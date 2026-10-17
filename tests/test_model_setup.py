import json
from pathlib import Path

import pytest

from ktts import cli
from ktts.model_setup import download_voices, setup_model_and_voices


def test_setup_downloads_model_config_and_requested_voices_atomically(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "hub"
    calls: list[str] = []

    def download(*, repo_id: str, filename: str) -> str:
        assert repo_id
        calls.append(filename)
        source = downloads / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            json.dumps({"model": "fixture"}) if filename == "config.json" else filename
        )
        return str(source)

    model_path, voices = setup_model_and_voices(
        ["af_heart", "bf_emma", "af_heart"],
        model_dir=tmp_path / "models",
        voices_dir=tmp_path / "voices",
        downloader=download,
    )

    assert model_path == tmp_path / "models/v1_0/kokoro-v1_0.pth"
    assert model_path.read_text() == "kokoro-v1_0.pth"
    assert json.loads((model_path.parent / "config.json").read_text()) == {
        "model": "fixture"
    }
    assert [voice.name for voice in voices] == ["af_heart.pt", "bf_emma.pt"]
    assert [voice.read_text() for voice in voices] == [
        "voices/af_heart.pt",
        "voices/bf_emma.pt",
    ]
    assert calls == [
        "kokoro-v1_0.pth",
        "config.json",
        "voices/af_heart.pt",
        "voices/bf_emma.pt",
    ]

    setup_model_and_voices(
        ["af_heart", "bf_emma"],
        model_dir=tmp_path / "models",
        voices_dir=tmp_path / "voices",
        downloader=download,
    )
    assert calls == [
        "kokoro-v1_0.pth",
        "config.json",
        "voices/af_heart.pt",
        "voices/bf_emma.pt",
    ]


def test_setup_rejects_invalid_voice_names_before_downloading(tmp_path: Path) -> None:
    def unexpected_download(**kwargs) -> str:
        raise AssertionError(f"Unexpected download: {kwargs}")

    with pytest.raises(ValueError, match="Voice names"):
        setup_model_and_voices(
            ["../../outside"],
            model_dir=tmp_path / "models",
            voices_dir=tmp_path / "voices",
            downloader=unexpected_download,
        )


def test_download_voices_fetches_only_voice_files(tmp_path: Path) -> None:
    calls: list[str] = []

    def download(*, repo_id: str, filename: str) -> str:
        calls.append(filename)
        source = tmp_path / "hub" / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(filename)
        return str(source)

    paths = download_voices(
        ["bf_emma"], voices_dir=tmp_path / "voices", downloader=download
    )

    assert paths == [tmp_path / "voices" / "bf_emma.pt"]
    assert calls == ["voices/bf_emma.pt"]
    download_voices(["bf_emma"], voices_dir=tmp_path / "voices", downloader=download)
    assert calls == ["voices/bf_emma.pt"]


def test_download_flag_downloads_voices_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requested: list[list[str]] = []

    def fake_download(voices: list[str]) -> list[Path]:
        requested.append(voices)
        return [Path(f"/voices/{voice}.pt") for voice in voices]

    monkeypatch.setattr(cli, "download_voices", fake_download)

    cli.main(["-d", "bf_emma", "--download", "am_adam"])

    assert requested == [["bf_emma", "am_adam"]]
    assert "/voices/bf_emma.pt" in capsys.readouterr().out


def test_download_flag_rejects_synthesis_input() -> None:
    with pytest.raises(SystemExit):
        cli.main(["-d", "bf_emma", "Hello."])


def test_setup_subcommand_uses_default_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[list[str]] = []
    monkeypatch.setattr(
        cli, "setup_model_and_voices", lambda voices: selected.append(voices)
    )

    cli.main(["setup"])

    assert selected == [["af_heart"]]


def test_setup_subcommand_accepts_repeated_voice_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[list[str]] = []
    monkeypatch.setattr(
        cli, "setup_model_and_voices", lambda voices: selected.append(voices)
    )

    cli.main(["setup", "--voice", "af_heart", "-v", "bf_emma"])

    assert selected == [["af_heart", "bf_emma"]]
