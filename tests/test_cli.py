from pathlib import Path

import pytest

from ktts import cli


class FakeSession:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.id = "a" * 32
        self.cursor = 0
        self.chunks = []
        self.state = "done"
        self.resumed = False

    def resume(self):
        self.resumed = True
        self.state = "running"

    async def stream(self):
        if False:
            yield None


def test_help_lists_phase_five_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--file" in output
    assert "--listvoices" in output
    assert "--resume" in output
    assert "--lexicon" in output


@pytest.mark.parametrize("speed", ["0.49", "1.51", "fast"])
def test_cli_rejects_invalid_speed(speed: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--speed", speed, "hello"])

    assert exit_info.value.code == 2


@pytest.mark.parametrize("speed", ["0.5", "1.5"])
def test_cli_accepts_speed_range_boundaries(speed: str) -> None:
    args = cli.build_parser().parse_args(["--speed", speed, "hello"])
    assert args.speed == float(speed)


def test_cli_rejects_both_positional_and_file_input(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello", encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["hello", "--file", str(text_file)])

    assert exit_info.value.code == 2


def test_output_path_rules(tmp_path: Path) -> None:
    input_file = tmp_path / "notes.txt"

    assert cli._resolve_output_path(None, input_file) == (
        Path("notes.wav").resolve(),
        None,
    )
    resolved, notice = cli._resolve_output_path(Path("speech.mp3"), None)
    assert resolved == Path("speech.wav").resolve()
    assert notice == "Only WAV output is available; writing speech.wav."
    assert cli._resolve_output_path(Path("speech.wav"), None) == (
        Path("speech.wav").resolve(),
        None,
    )


def test_output_folder_gets_default_file_name(tmp_path: Path) -> None:
    folder = tmp_path / "audio"
    folder.mkdir()
    input_file = tmp_path / "notes.txt"

    assert cli._resolve_output_path(str(folder), input_file) == (
        (folder / "notes.wav").resolve(),
        None,
    )
    # A trailing slash marks a folder that does not exist yet.
    assert cli._resolve_output_path(f"{tmp_path}/new/", input_file) == (
        (tmp_path / "new" / "notes.wav").resolve(),
        None,
    )
    resolved, notice = cli._resolve_output_path(str(folder), None)
    assert resolved.parent == folder.resolve()
    assert resolved.suffix == ".wav" and notice is None


def test_lexicon_parser_applies_supported_ipa_substitution(tmp_path: Path) -> None:
    lexicon_file = tmp_path / "names.txt"
    lexicon_file.write_text("Name = nᵊm\n", encoding="utf-8")

    assert cli._load_lexicon(lexicon_file) == {"Name": "nəm"}


def test_lexicon_parser_reports_unsupported_symbols(tmp_path: Path) -> None:
    lexicon_file = tmp_path / "names.txt"
    lexicon_file.write_text("Name = ☃\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported IPA symbols"):
        cli._load_lexicon(lexicon_file)


def test_cli_creates_session_with_validated_voice_and_wav_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("Say Quenthariel.", encoding="utf-8")
    created = {}

    async def list_voices():
        return ["af_heart"]

    async def create(text: str, **kwargs):
        created["text"] = text
        created.update(kwargs)
        return FakeSession(Path(kwargs["output_path"]))

    monkeypatch.setattr(cli.paths, "list_voices", list_voices)
    monkeypatch.setattr(cli.Session, "create", create)

    cli.main(["--file", str(text_file), "--lexicon", str(_write_lexicon(tmp_path))])

    assert created["text"] == "Say Quenthariel."
    assert created["voice"] == "af_heart"
    assert created["speed"] == 0.9
    assert created["lexicon"] == {"Quenthariel": "kwɛnθɑɹiəl"}
    assert created["output_path"] == Path("notes.wav").resolve()
    assert "chunk" not in capsys.readouterr().out


def _write_lexicon(directory: Path) -> Path:
    lexicon = directory / "names.txt"
    lexicon.write_text("Quenthariel = kwɛnθɑɹiəl\n", encoding="utf-8")
    return lexicon


def test_cli_refuses_existing_output_without_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "speech.wav"
    output.write_bytes(b"existing")

    async def list_voices():
        return ["af_heart"]

    monkeypatch.setattr(cli.paths, "list_voices", list_voices)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["hello", "--output", str(output)])

    assert exit_info.value.code == 2


def test_cli_force_allows_existing_timestamp_output_for_positional_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "speech.wav"
    output.write_bytes(b"existing")
    created = {}

    async def list_voices():
        return ["af_heart"]

    async def create(text: str, **kwargs):
        created["text"] = text
        created.update(kwargs)
        return FakeSession(Path(kwargs["output_path"]))

    monkeypatch.setattr(cli, "_resolve_output_path", lambda *_: (output, None))
    monkeypatch.setattr(cli.paths, "list_voices", list_voices)
    monkeypatch.setattr(cli.Session, "create", create)

    cli.main(["hello", "--force"])

    assert created["text"] == "hello"
    assert created["output_path"] == output


def test_listvoices_prints_names(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def list_voices():
        return ["af_heart", "am_adam"]

    monkeypatch.setattr(cli.paths, "list_voices", list_voices)

    cli.main(["--listvoices"])

    assert capsys.readouterr().out.splitlines() == ["af_heart", "am_adam"]


def test_sessions_option_prints_saved_session_progress(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "list_saved_sessions",
        lambda: [
            {
                "id": "c" * 32,
                "state": "paused",
                "cursor": 2,
                "total": 5,
                "updated_at": "2026-10-15T12:00:00+00:00",
            }
        ],
    )

    cli.main(["--sessions"])

    assert "paused  chunk 2/5" in capsys.readouterr().out


def test_resume_without_id_uses_most_recent_unfinished(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    restored = {}
    session = FakeSession(Path("saved.wav"))

    def latest_id():
        return "b" * 32

    async def restore(session_id: str):
        restored["id"] = session_id
        return session

    monkeypatch.setattr(cli, "latest_unfinished_session_id", latest_id)
    monkeypatch.setattr(cli.Session, "restore", restore)

    cli.main(["--resume"])

    assert restored["id"] == "b" * 32
    assert session.resumed
    assert "saved.wav" in capsys.readouterr().out
