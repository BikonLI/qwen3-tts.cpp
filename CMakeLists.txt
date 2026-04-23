cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

# Timing instrumentation option
option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

# Find threads
find_package(Threads REQUIRED)

# GGML library paths
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
set(GGML_BUILD_DIR "${GGML_DIR}/build")
# set(SDL3_DIR "${CMAKE_SOURCE_DIR}/cmake")
find_package(SDL3 REQUIRED)

if(WIN32)
    set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
    add_library(imgui STATIC
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_win32.cpp
        ${IMGUI_DIR}/backends/imgui_impl_dx11.cpp
    )
    target_include_directories(imgui PUBLIC
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
    )
    target_link_libraries(imgui PUBLIC d3d11 dxgi d3dcompiler)
endif()


# Detect Metal backend availability
if(APPLE AND EXISTS "${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.dylib")
    set(GGML_HAS_METAL ON)
    message(STATUS "GGML Metal backend found — GPU acceleration enabled")
else()
    set(GGML_HAS_METAL OFF)
endif()

# Single shared library (C API + C++ symbols)
set(QWEN3TTS_SHARED_SOURCES
    src/core/gguf_loader.cpp
    src/core/text_tokenizer.cpp
    src/core/tts_transformer.cpp
    src/audio/audio_tokenizer_encoder.cpp
    src/audio/audio_tokenizer_decoder.cpp
    src/audio/audio_codec_encoder.cpp
    src/qwen3_tts.cpp
    src/api/qwen3tts_c_api.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND QWEN3TTS_SHARED_SOURCES src/core/coreml_code_predictor.mm)
    set_source_files_properties(src/core/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND QWEN3TTS_SHARED_SOURCES src/core/coreml_code_predictor_stub.cpp)
endif()

add_library(qwen3tts_shared SHARED
    ${QWEN3TTS_SHARED_SOURCES}
)
set_target_properties(qwen3tts_shared PROPERTIES
    OUTPUT_NAME "qwen3tts"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
if(WIN32)
    # Export both C API and C++ symbols for DLL loading and import-lib usage.
    set_target_properties(qwen3tts_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
target_include_directories(qwen3tts_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
    ${GGML_DIR}/include
)
target_link_directories(qwen3tts_shared PUBLIC
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/ggml-metal
)
target_link_libraries(qwen3tts_shared PUBLIC
    ggml
    ggml-base
    ggml-cpu
    SDL3::SDL3
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(qwen3tts_shared PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(qwen3tts_shared PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()
if(WIN32)
    target_link_libraries(qwen3tts_shared PUBLIC Psapi)
endif()

# CLI executable
add_executable(qwen3-tts-cli
    src/app/main.cpp
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3tts_shared
)

if(WIN32)
    add_executable(qwen3-tts-gui WIN32
        src/app/gui_main.cpp
    )
    if(MSVC)
        target_compile_options(qwen3-tts-gui PRIVATE /utf-8)
    endif()
    target_link_libraries(qwen3-tts-gui PRIVATE
        qwen3tts_shared
        imgui
        Comdlg32
    )
endif()

# Test executable for text8 tokenization comparison
add_executable(test_text8_tokenize
    tests/test_text8_tokenize.cpp
)
target_link_libraries(test_text8_tokenize PRIVATE
    qwen3tts_shared
)
if(MSVC)
    target_compile_options(test_text8_tokenize PRIVATE /utf-8)
endif()

# Install targets
install(TARGETS qwen3tts_shared qwen3-tts-cli
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
if(WIN32)
    install(TARGETS qwen3-tts-gui
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()
install(FILES
    include/qwen3_tts.h
    include/qwen3tts_c_api.h
    DESTINATION include
)
