#!/usr/bin/env python3
"""
Compatibility entry for older docs/commands.

The main converter now includes all v2 metadata/projection handling,
so this script forwards directly to convert_tts_to_gguf.py.
"""

from convert_tts_to_gguf import main


if __name__ == "__main__":
    main()
