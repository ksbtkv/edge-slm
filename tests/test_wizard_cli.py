from __future__ import annotations

from unittest.mock import patch

from wizard.cli import main
from wizard.model_catalog import CATALOG


def test_main_returns_1_when_ollama_missing():
    with patch("wizard.cli.is_ollama_installed", return_value=False):
        assert main(["--model", CATALOG[0].name]) == 1


def test_main_pulls_and_runs_chosen_model():
    with patch("wizard.cli.is_ollama_installed", return_value=True), patch(
        "wizard.cli.pull"
    ) as mock_pull, patch("wizard.cli.run_interactive", return_value=0) as mock_run:
        code = main(["--model", CATALOG[0].name])

        mock_pull.assert_called_once_with(CATALOG[0].ollama_ref)
        mock_run.assert_called_once_with(CATALOG[0].ollama_ref)
        assert code == 0


def test_main_propagates_run_interactive_exit_code():
    with patch("wizard.cli.is_ollama_installed", return_value=True), patch(
        "wizard.cli.pull"
    ), patch("wizard.cli.run_interactive", return_value=7):
        assert main(["--model", CATALOG[0].name]) == 7
