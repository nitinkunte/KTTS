import asyncio

import pytest

from ktts import __version__
from ktts.cli import main
from ktts.model_manager import ModelManager


def test_package_bootstraps_with_expected_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_help_is_available(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["ktts", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 0
    assert "usage: ktts" in capsys.readouterr().out


def test_missing_model_does_not_exit_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ModelManager()

    async def initialize() -> None:
        return None

    async def fail_to_load(_: str) -> None:
        raise FileNotFoundError("missing model")

    monkeypatch.setattr(manager, "initialize", initialize)
    monkeypatch.setattr(manager, "load_model", fail_to_load)

    with pytest.raises(FileNotFoundError, match="missing model"):
        asyncio.run(manager.ensure_backend())
