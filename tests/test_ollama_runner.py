from __future__ import annotations

from unittest.mock import MagicMock, patch

from wizard.ollama_runner import is_ollama_installed, pull, run_interactive


def test_is_ollama_installed_true_when_on_path():
    with patch("wizard.ollama_runner.shutil.which", return_value="/usr/local/bin/ollama"):
        assert is_ollama_installed() is True


def test_is_ollama_installed_false_when_missing():
    with patch("wizard.ollama_runner.shutil.which", return_value=None):
        assert is_ollama_installed() is False


def test_pull_invokes_ollama_pull():
    with patch("wizard.ollama_runner.subprocess.run") as mock_run:
        pull("qwen3:4b-instruct-2507")
        mock_run.assert_called_once_with(
            ["ollama", "pull", "qwen3:4b-instruct-2507"], check=True
        )


def test_run_interactive_invokes_ollama_run_and_returns_code():
    with patch("wizard.ollama_runner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        code = run_interactive("qwen3:4b-instruct-2507")
        mock_run.assert_called_once_with(["ollama", "run", "qwen3:4b-instruct-2507"])
        assert code == 0
