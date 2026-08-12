"""
Try It: the simplest Entry Point (see `CONTEXT.md`) — no data, no
fine-tuning, just install and run a stock Model Catalog entry to sanity-check
it on this machine.
"""

from __future__ import annotations

import argparse
import sys

from wizard.model_catalog import CATALOG, get_entry
from wizard.ollama_runner import is_ollama_installed, pull, run_interactive


def prompt_for_entry() -> str:
    print("Pick a model to try:\n")
    for i, entry in enumerate(CATALOG, start=1):
        print(f"  {i}. {entry.display_name} (~{entry.approx_download_gb} GB)")
        print(f"     {entry.description}")
    print()

    while True:
        choice = input(f"Enter a number (1-{len(CATALOG)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(CATALOG):
            return CATALOG[int(choice) - 1].name
        print(f"Please enter a number between 1 and {len(CATALOG)}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=[entry.name for entry in CATALOG],
        help="Model Catalog entry to run (skips the interactive picker)",
    )
    args = parser.parse_args(argv)

    if not is_ollama_installed():
        print(
            "Ollama isn't installed. Install it from https://ollama.com/download "
            "and run this again.",
            file=sys.stderr,
        )
        return 1

    entry = get_entry(args.model) if args.model else get_entry(prompt_for_entry())

    print(f"\nInstalling {entry.display_name} ({entry.ollama_ref})...")
    pull(entry.ollama_ref)

    print(f"\nRunning {entry.display_name}. Type /bye to exit.\n")
    return run_interactive(entry.ollama_ref)


if __name__ == "__main__":
    raise SystemExit(main())
