"""
Thin wrapper around the `ollama` CLI for the Try It entry point (see
`CONTEXT.md`): install and run a stock Model Catalog entry as-is, no
fine-tuning involved. Stdlib-only, like the rest of the base install.
"""

from __future__ import annotations

import shutil
import subprocess


def is_ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def pull(ollama_ref: str) -> None:
    subprocess.run(["ollama", "pull", ollama_ref], check=True)


def run_interactive(ollama_ref: str) -> int:
    result = subprocess.run(["ollama", "run", ollama_ref])
    return result.returncode
