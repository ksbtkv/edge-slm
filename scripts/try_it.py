#!/usr/bin/env python
"""
Wizard Try It: install and run a stock Model Catalog model locally via
Ollama, no fine-tuning or data required.

Examples:

    # Interactive picker
    python scripts/try_it.py

    # Skip the picker
    python scripts/try_it.py --model qwen3-4b
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wizard.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
