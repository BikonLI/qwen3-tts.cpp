<div align="center">
  <h1>qwen3-tts.cpp</h1>
  <p><strong>Qwen3-TTS 全系模型 · 全功能 · C++17 工业级部署方案</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D4?style=for-the-badge" alt="platform" />
    <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge" alt="cxx17" />
    <img src="https://img.shields.io/badge/CMake-3.14%2B-064F8C?style=for-the-badge" alt="cmake" />
    <img src="https://img.shields.io/badge/Backend-ggml-222222?style=for-the-badge" alt="ggml" />
    <img src="https://img.shields.io/badge/Audio-SDL3-7A4DFF?style=for-the-badge" alt="sdl3" />
    <img src="https://img.shields.io/badge/Output-DLL%20%2B%20LIB%20%2B%20EXE-00A86B?style=for-the-badge" alt="outputs" />
  </p>
  <p>
    <a href="#highlights">项目亮点</a> ·
    <a href="#windows-build">Windows 构建</a> ·
    <a href="#artifacts">产物说明</a> ·
    <a href="#cli">CLI 示例</a> ·
    <a href="#integration">DLL 与 C++ 集成</a>
  </p>
</div>

<div align="center">
  <img src="./docs/benchmark_pytorch_vs_cpp.png" alt="benchmark" width="920" />
</div>

<hr />

<a id="highlights"></a>

## 项目亮点

qwen3-tts.cpp 是面向真实部署场景打造的 Qwen3-TTS C++ 全链路实现。

全系模型、全任务形态、全流程推理，一套工程打通。

这不是一个只能跑 Demo 的仓库，而是一套可直接进入工程体系的完整方案。可以非常自信地说：在 Qwen3-TTS 的 C++ 落地方向上，它是目前最全面、最完整、最工程化、最实战的方案之一。

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>全模型覆盖</h3>
      <p>0.6B / 1.7B，Base / CustomVoice / VoiceDesign 全支持。</p>
    </td>
    <td width="33%" valign="top">
      <h3>全功能覆盖</h3>
      <p>基础 TTS、语音克隆、x-vector-only、speaker/instruct、多语言控制全打通。</p>
    </td>
    <td width="33%" valign="top">
      <h3>全链路部署</h3>
      <p>Tokenizer、Speaker Encoder、Transformer、Vocoder 纯 C++17 推理。</p>
    </td>
  </tr>
</table>

## 支持矩阵

来源：<a href="https://qwen.ai/blog?id=qwen3tts-0115">Qwen 官方博客（Qwen3-TTS）</a>

### 1.7B 模型

| 模型 | 功能 | 语种支持 | 流式生成 | 指令控制 |
| --- | --- | --- | --- | --- |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 根据用户输入描述进行音色创造。 | 中、英、日、韩、德、法、俄、葡萄牙、西班牙、意大利 | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 根据用户输入指令对目标音色进行风格控制；支持 9 个精品音色，涵盖多种性别、年龄、语种与方言组合。 | 中、英、日、韩、德、法、俄、葡萄牙、西班牙、意大利 | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-Base | 基座模型，可以根据用户输入音频进行 3s 快速音色克隆，可用于 FT 到其他模型。 | 中、英、日、韩、德、法、俄、葡萄牙、西班牙、意大利 | ✅ | — |

### 0.6B 模型

| 模型 | 功能 | 语种支持 | 流式生成 | 指令控制 |
| --- | --- | --- | --- | --- |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 支持 9 个精品音色，涵盖多种性别、年龄、语种与方言组合。 | 中、英、日、韩、德、法、俄、葡萄牙、西班牙、意大利 | ✅ | — |
| Qwen3-TTS-12Hz-0.6B-Base | 基座模型，可以根据用户输入音频进行 3s 快速音色克隆，可用于 FT 到其他模型。 | 中、英、日、韩、德、法、俄、葡萄牙、西班牙、意大利 | ✅ | — |

## 依赖与环境

本项目核心依赖：

- ggml
- SDL3

推荐环境：

- Windows 10/11 x64
- Visual Studio 2022（MSVC）
- CMake 3.14+
- Python（仅用于模型下载/转换脚本，推理运行时不依赖 Python）

<a id="windows-build"></a>

## Windows 构建（重点）

<details open>
  <summary><strong>Step 1: 获取源码</strong></summary>

```powershell
git clone https://github.com/BikonLI/qwen3-tts.cpp
cd qwen3-tts.cpp
git submodule update --init --recursive
```

</details>

<details open>
  <summary><strong>Step 2: 先构建 ggml</strong></summary>

```powershell
cmake -S ggml -B ggml/build -G "Visual Studio 17 2022"
cmake --build ggml/build --config Release -j 8
```

说明：顶层工程默认从 ./ggml/build/src 链接 ggml 产物。

</details>

<details open>
  <summary><strong>Step 3: 准备 SDL3（确保 SDL3Config.cmake 可发现）</strong></summary>

SDL3 常见目录结构示例：

```text
.
│  .git-hash
│  INSTALL.md
│  LICENSE.txt
│  README.md
│
├─cmake
│      SDL3Config.cmake
│      SDL3ConfigVersion.cmake
│      sdlcpu.cmake
│
├─include
│  └─SDL3
│          SDL.h
│          SDL_assert.h
│          SDL_asyncio.h
│          SDL_atomic.h
│          SDL_audio.h
```

</details>

<details open>
  <summary><strong>Step 4: 配置并构建主工程</strong></summary>

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -DSDL3_DIR="D:/example_dir/SDL3-3.4.4/cmake"
cmake --build build --config Release -j 8
```

</details>

<details>
  <summary><strong>Step 5: 获取模型（可选但推荐）</strong></summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install huggingface_hub gguf torch safetensors numpy tqdm coremltools
python scripts/setup_pipeline_models.py
```

说明：该脚本会下载并转换 Qwen3-TTS 家族模型，也可传入参数进行精细控制。

</details>

<details>
  <summary><strong>Step 6: 安装到 dist（可选）</strong></summary>

```powershell
cmake --install build --config Release --prefix dist
```

安装后典型目录：

- dist/bin/qwen3tts.dll
- dist/bin/qwen3-tts-cli.exe
- dist/lib/qwen3tts.lib
- dist/include/qwen3_tts.h
- dist/include/qwen3tts_c_api.h

</details>

<a id="artifacts"></a>

## 构建产物说明

在默认 Release 构建下，核心输出位于：

- build/Release/qwen3tts.dll
- build/Release/qwen3tts.lib
- build/Release/qwen3-tts-cli.exe

<table>
  <tr>
    <th>产物</th>
    <th>定位</th>
    <th>说明</th>
  </tr>
  <tr>
    <td>qwen3tts.dll</td>
    <td>运行时核心库</td>
    <td>可通过 LoadLibrary/GetProcAddress 动态加载，适合插件化和跨语言调用。</td>
  </tr>
  <tr>
    <td>qwen3tts.lib</td>
    <td>导入库</td>
    <td>用于链接 DLL 导出的符号，配合头文件可直接走 C++ API。</td>
  </tr>
  <tr>
    <td>qwen3-tts-cli.exe</td>
    <td>命令行工具</td>
    <td>开箱即用，适合验证模型、批处理生成和集成测试。</td>
  </tr>
</table>

注意：qwen3tts.lib 是 DLL 的导入库，不是独立静态实现库。

<a id="cli"></a>

## CLI 快速上手

关键参数：

- -m：功能模型路径或目录（Base/CustomVoice/VoiceDesign）
- -mt：tokenizer/vocoder 模型路径或目录
- -t：输入文本（也可传文本文件）

注意：PowerShell 下中文参数可能受编码影响，中文文本建议通过文件输入。

<details>
  <summary><strong>查看完整示例命令（覆盖全任务）</strong></summary>

```powershell
--------------------------------------------------
1) 0.6B Base - 声音克隆（ref + ref-text）
--------------------------------------------------
.\build\Release\qwen3-tts-cli.exe -m .\models\gguf\0.6b-base\qwen3-tts-12hz-0.6b-base-f16.gguf -mt .\models\gguf\tokenizer\qwen3-tts-tokenizer-12hz-f16.gguf -t "Hello, this is a 0.6B base voice clone test." -r .\examples\readme_clone_input.wav --ref-text "Hello, this is a short reference transcript." -o .\outputs\test_06b_base_clone.wav

--------------------------------------------------
2) 0.6B Base - x-vector only 克隆（无需 ref-text）
--------------------------------------------------
.\build\Release\qwen3-tts-cli.exe -m .\models\gguf\0.6b-base\qwen3-tts-12hz-0.6b-base-f16.gguf -mt .\models\gguf\tokenizer\qwen3-tts-tokenizer-12hz-f16.gguf -t "Hello, this is a 0.6B x-vector-only clone test." -r .\examples\readme_clone_input.wav --x-vector-only -o .\outputs\test_06b_base_xvector_only.wav

--------------------------------------------------
3) 0.6B CustomVoice - 预置音色（无 instruct）
--------------------------------------------------
.\build\Release\qwen3-tts-cli.exe -m .\models\gguf\0.6b-custom-voice\qwen3-tts-12hz-0.6b-custom-voice-f16.gguf -mt .\models\gguf\tokenizer\qwen3-tts-tokenizer-12hz-f16.gguf -t "This is a 0.6B custom voice test." --speaker ryan -o .\outputs\test_06b_custom_voice_ryan.wav

--------------------------------------------------
4) 1.7B Base - 声音克隆（ref + ref-text）
--------------------------------------------------
.\build\Release\qwen3-tts-cli.exe -m .\models\gguf\1.7b-base\qwen3-tts-12hz-1.7b-base-f16.gguf -mt .\models\gguf\tokenizer\qwen3-tts-tokenizer-12hz-f16.gguf -t "Hello, this is a 1.7B base voice clone test." -r .\examples\readme_clone_input.wav --ref-text "Hello, this is a short reference transcript." -o .\outputs\test_17b_base_clone.wav

--------------------------------------------------
5) 1.7B CustomVoice - 预置音色（无 instruct）
--------------------------------------------------
.\build\Release\qwen3-tts-cli.exe -m .\models\gguf\1.7b-custom-voice\qwen3-tts-12hz-1.7b-custom-voice-f16.gguf -mt .\models\gguf\tokenizer\qwen3-tts-tokenizer-12hz-f16.gguf -t "This is a 1.7B custom voice test." --speaker serena -o .\outputs\test_17b_custom_voice_serena.wav

--------------------------------------------------
6) 1.7B CustomVoice - 预置音色 + instruct
--------------------------------------------------
.\build\Release\qwen3-tts-cli.exe -m .\models\gguf\1.7b-custom-voice\qwen3-tts-12hz-1.7b-custom-voice-f16.gguf -mt .\models\gguf\tokenizer\qwen3-tts-tokenizer-12hz-f16.gguf -t "This is a 1.7B custom voice plus instruct test." --speaker ryan --instruct "Speak with strong excitement and a bright tone." -o .\outputs\test_17b_custom_voice_ryan_instruct.wav

--------------------------------------------------
7) 1.7B VoiceDesign - instruct 设计音色
--------------------------------------------------
.\build\Release\qwen3-tts-cli.exe -m .\models\gguf\1.7b-voice-design\qwen3-tts-12hz-1.7b-voice-design-f16.gguf -mt .\models\gguf\tokenizer\qwen3-tts-tokenizer-12hz-f16.gguf -t "This is a 1.7B voice design test." --instruct "Female voice, warm timbre, calm and reassuring." -o .\outputs\test_17b_voice_design_warm.wav
.\build\Release\qwen3-tts-cli.exe -m .\models\gguf\1.7b-voice-design\qwen3-tts-12hz-1.7b-voice-design-f16.gguf -mt .\models\gguf\tokenizer\qwen3-tts-tokenizer-12hz-f16.gguf -t "This is a 1.7B voice design test." --instruct "male voice, warm timbre, calm and reassuring." -o .\outputs\test_17b_voice_design_warm.wav

--------------------------------------------------
8) 1.7B VoiceDesign - 中文 + 语言指定
--------------------------------------------------
.\build\Release\qwen3-tts-cli.exe -m .\models\gguf\1.7b-voice-design\qwen3-tts-12hz-1.7b-voice-design-f16.gguf -mt .\models\gguf\tokenizer\qwen3-tts-tokenizer-12hz-f16.gguf -t "这是一个中文语音设计测试。" --instruct "女声，温柔，语速稍慢，情绪稳定。" -l zh -o .\outputs\test_17b_voice_design_zh.wav
```

</details>

<a id="integration"></a>

## DLL 与 C++ 集成

### 方式 A：无头文件动态加载（LoadLibrary）

qwen3tts.dll 对外导出 C API，典型符号包括：

- qwen3_tts_create
- qwen3_tts_default_params
- qwen3_tts_synthesize
- qwen3_tts_synthesize_with_voice_file
- qwen3_tts_extract_embedding_file
- qwen3_tts_synthesize_with_embedding
- qwen3_tts_free_audio
- qwen3_tts_destroy

通过 GetProcAddress 绑定函数指针即可完成“无头文件硬依赖”调用。

### 方式 B：C++ 头文件 + 导入库

如果你希望直接调用 C++ 符号：

1. 链接 qwen3tts.lib
2. 包含 include/qwen3_tts.h
3. 运行时放置 qwen3tts.dll

```cpp
#include "qwen3_tts.h"

int main() {
    qwen3_tts::Qwen3TTS tts;
    if (!tts.load_models_from_files(
            "models/gguf/Qwen3-TTS-12Hz-1.7B-Base/model.gguf",
            "models/gguf/Qwen3-TTS-Tokenizer-12Hz/tokenizer.gguf")) {
        return 1;
    }

    qwen3_tts::tts_result r = tts.synthesize("Hello from C++ API");
    return r.success ? 0 : 2;
}
```

## 为什么它是最完整的 Qwen3-TTS C++ 部署方案

- 一套工程同时覆盖 CLI、C API、C++ API 三种接入层。
- 一套权重体系覆盖 Base、CustomVoice、VoiceDesign 全任务族。
- 一套构建配置直接产出 DLL + LIB + EXE，部署结构清晰。
- 一套核心代码完成 tokenizer、speaker encoder、transformer、vocoder 全链路推理。
- 一套 Windows 优先的工程实践，真正对部署友好。

一句话总结：

如果你要的是“能跑、好接、可控、可扩展、可上线”的 Qwen3-TTS C++ 落地方案，qwen3-tts.cpp 就是你要找的版本。

## 致谢

- Qwen3-TTS by Alibaba Qwen Team
- ggml by ggml-org
- WavTokenizer 相关工作
- 原始仓库: https://github.com/predict-woo/qwen3-tts.cpp

## GitHub Star 趋势图

如果喜欢，那么给我一个**Star**!
<a href="https://star-history.com/#BikonLI/qwen3-tts.cpp&Date">
  <img alt="GitHub Star History Chart" src="https://api.star-history.com/svg?repos=BikonLI/qwen3-tts.cpp&type=Date" />
</a>

