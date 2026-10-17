import asyncio
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from ktts.base import AudioChunk
from ktts.engine import PlannedChunk
from ktts.session import (
    Session,
    SessionError,
    SessionExpiredError,
    SessionLockedError,
    SessionPlanMismatchError,
)


def _plan(*texts: str) -> list[PlannedChunk]:
    return [
        PlannedChunk(text=text, tokens=(1,), phonemes="həˈloʊ")
        for text in texts
    ]


def _fake_renderer():
    async def render(_: PlannedChunk):
        yield AudioChunk(np.full(2048, 0.25, dtype=np.float32))

    return render


def _install_session_fakes(monkeypatch: pytest.MonkeyPatch, texts: list[str]) -> None:
    async def build_plan(*args, **kwargs):
        return _plan(*texts)

    async def get_voice_path(_: str) -> str:
        return "/fake/af_heart.pt"

    monkeypatch.setattr("ktts.session.plan", build_plan)
    monkeypatch.setattr("ktts.session.paths.get_voice_path", get_voice_path)


def test_session_pause_resume_and_persisted_restore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_session_fakes(monkeypatch, ["Hello.", "Goodbye."])

    async def run():
        session = await Session.create(
            "Hello. Goodbye.",
            session_root=tmp_path,
            renderer=_fake_renderer(),
        )
        stream = session.stream()
        first = await anext(stream)
        assert first.audio.dtype == np.int16
        assert session.cursor == 1

        session.pause()
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        assert session.state == "paused"
        session.resume()
        second = await pending
        assert second.audio.dtype == np.int16
        assert session.state == "done"
        await stream.aclose()
        return session

    session = asyncio.run(run())

    assert session.state_path.is_file()
    assert session.output_path.is_file()
    with wave.open(str(session.output_path), "rb") as output:
        assert output.getframerate() == 24000
        assert output.getnframes() > 0

    async def restore():
        return await Session.restore(
            session.id,
            session_root=tmp_path,
            renderer=_fake_renderer(),
        )

    restored = asyncio.run(restore())
    assert restored.cursor == 2
    assert restored.state == "done"


def test_selected_output_path_survives_session_restore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_session_fakes(monkeypatch, ["Hello."])
    selected_output = tmp_path / "exports" / "hello.wav"

    async def create():
        return await Session.create(
            "Hello.",
            session_root=tmp_path / "sessions",
            output_path=selected_output,
            renderer=_fake_renderer(),
        )

    session = asyncio.run(create())

    async def restore():
        return await Session.restore(
            session.id,
            session_root=tmp_path / "sessions",
            renderer=_fake_renderer(),
        )

    restored = asyncio.run(restore())
    assert restored.output_path == selected_output


def test_paused_session_does_not_block_another_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_session_fakes(monkeypatch, ["Hello."])

    async def run():
        paused = await Session.create(
            "Hello.", session_root=tmp_path / "paused", renderer=_fake_renderer()
        )
        paused.pause()
        running = await Session.create(
            "Hello.", session_root=tmp_path / "running", renderer=_fake_renderer()
        )

        async def consume(session: Session) -> None:
            async for _ in session.stream():
                pass

        running_task = asyncio.create_task(consume(running))
        await asyncio.wait_for(running_task, timeout=3)
        assert running.state == "done"
        assert paused.state == "paused"
        assert paused.cursor == 0

        paused.resume()
        await consume(paused)
        assert paused.state == "done"

    asyncio.run(run())


def test_restore_recovers_chunk_written_before_cursor_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_session_fakes(monkeypatch, ["One.", "Two."])

    async def create():
        return await Session.create("One. Two.", session_root=tmp_path)

    session = asyncio.run(create())
    session.writer.write_chunk(0, np.zeros(64, dtype=np.int16))
    metadata = json.loads(session.state_path.read_text(encoding="utf-8"))
    assert metadata["cursor"] == 0

    async def restore():
        return await Session.restore(session.id, session_root=tmp_path)

    restored = asyncio.run(restore())
    assert restored.cursor == 1
    assert restored.writer.chunk_path(0).is_file()


def test_restore_discards_partial_chunk_and_resumes_from_its_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_session_fakes(monkeypatch, ["One.", "Two."])

    async def create():
        return await Session.create("One. Two.", session_root=tmp_path)

    session = asyncio.run(create())
    session.writer.chunk_path(0).write_bytes(b"partial wav")

    async def restore():
        return await Session.restore(
            session.id,
            session_root=tmp_path,
            renderer=_fake_renderer(),
        )

    restored = asyncio.run(restore())
    assert restored.cursor == 0
    assert not restored.writer.chunk_path(0).exists()


def test_plan_mismatch_requires_explicit_restart_or_replan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def initial_plan(*args, **kwargs):
        return _plan("Before.", "After.")

    async def get_voice_path(_: str) -> str:
        return "/fake/af_heart.pt"

    monkeypatch.setattr("ktts.session.plan", initial_plan)
    monkeypatch.setattr("ktts.session.paths.get_voice_path", get_voice_path)

    async def create():
        return await Session.create(
            "Before. After.",
            session_root=tmp_path,
            renderer=_fake_renderer(),
        )

    session = asyncio.run(create())
    session.writer.write_chunk(0, np.zeros(64, dtype=np.int16))
    session.cursor = 1
    session.state = "paused"
    session._save_metadata()

    async def changed_plan(text: str, *args, **kwargs):
        if text == "After.":
            return _plan("Replanned.")
        return _plan("Before.", "Replanned.")

    monkeypatch.setattr("ktts.session.plan", changed_plan)

    async def restore_without_choice():
        await Session.restore(session.id, session_root=tmp_path)

    with pytest.raises(SessionPlanMismatchError, match="on_plan_mismatch"):
        asyncio.run(restore_without_choice())

    async def restore_by_replanning():
        return await Session.restore(
            session.id,
            session_root=tmp_path,
            on_plan_mismatch="replan",
            renderer=_fake_renderer(),
        )

    replanned = asyncio.run(restore_by_replanning())
    assert replanned.cursor == 1
    assert [chunk.text for chunk in replanned.chunks] == ["Before.", "Replanned."]
    assert replanned.writer.chunk_path(0).is_file()
    assert not replanned.writer.chunk_path(1).exists()


def test_strict_plan_rejects_unmapped_phonemes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def split(*args, **kwargs):
        yield "Snowman.", [1], None

    monkeypatch.setattr("ktts.engine.smart_split", split)
    monkeypatch.setattr("ktts.engine.phonemize", lambda *_: "snow ☃")

    async def build_strict_plan():
        from ktts.engine import plan

        await plan("Snowman.", strict=True)

    with pytest.raises(ValueError, match="unmapped symbols.*☃"):
        asyncio.run(build_strict_plan())


def test_pause_tags_are_persisted_as_silence_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def build_plan(*args, **kwargs):
        return [
            PlannedChunk(text="", tokens=(), phonemes="", pause_duration_s=0.1),
        ]

    async def get_voice_path(_: str) -> str:
        return "/fake/af_heart.pt"

    monkeypatch.setattr("ktts.session.plan", build_plan)
    monkeypatch.setattr("ktts.session.paths.get_voice_path", get_voice_path)

    async def run():
        session = await Session.create("Hello [pause:0.1s] goodbye.", session_root=tmp_path)
        audio = [chunk async for chunk in session.stream()]
        return session, audio

    session, chunks = asyncio.run(run())
    assert session.state == "done"
    assert len(chunks) == 1
    assert len(chunks[0].audio) == 2400
    assert not np.any(chunks[0].audio)


def test_session_seek_and_stop_export_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_session_fakes(monkeypatch, ["One.", "Two."])

    async def create():
        return await Session.create("One. Two.", session_root=tmp_path)

    session = asyncio.run(create())
    with pytest.raises(ValueError, match="between 0 and 2"):
        session.seek(3)
    session.writer.write_chunk(0, np.zeros(32, dtype=np.int16))
    session.cursor = 1
    session.seek(0)
    assert session.cursor == 0
    assert not session.writer.chunk_path(0).exists()
    session.stop()
    assert session.state == "stopped"
    assert session.output_path.is_file()


def test_closing_stream_pauses_and_stop_exports_completed_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_session_fakes(monkeypatch, ["One.", "Two.", "Three."])

    async def run():
        session = await Session.create(
            "One. Two.",
            session_root=tmp_path,
            renderer=_fake_renderer(),
        )
        stream = session.stream()
        await anext(stream)
        await stream.aclose()
        assert session.state == "paused"
        assert session.cursor == 1

        session.resume()
        stream = session.stream()
        await anext(stream)
        session.stop()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        return session

    session = asyncio.run(run())
    assert session.state == "stopped"
    assert session.output_path.is_file()
    with wave.open(str(session.output_path), "rb") as output:
        assert output.getnframes() > 0

    async def restore():
        return await Session.restore(
            session.id,
            session_root=tmp_path,
            renderer=_fake_renderer(),
        )

    restored = asyncio.run(restore())
    assert restored.state == "paused"
    restored.resume()
    assert restored.state == "running"


def test_expired_session_is_rejected_without_deleting_its_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_session_fakes(monkeypatch, ["Hello."])

    async def create():
        return await Session.create("Hello.", session_root=tmp_path)

    session = asyncio.run(create())
    metadata = json.loads(session.state_path.read_text(encoding="utf-8"))
    metadata["updated_at"] = "2000-01-01T00:00:00+00:00"
    session.state_path.write_text(json.dumps(metadata), encoding="utf-8")

    async def restore():
        await Session.restore(session.id, session_root=tmp_path)

    with pytest.raises(SessionExpiredError):
        asyncio.run(restore())
    assert session.state_path.is_file()


def test_file_lock_and_concurrent_session_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_session_fakes(monkeypatch, ["Hello."])
    monkeypatch.setattr("ktts.session.settings.max_concurrent_sessions", 1)

    async def create():
        first = await Session.create("Hello.", session_root=tmp_path)
        second = await Session.create("Hello.", session_root=tmp_path)
        await first._register_active()
        try:
            with pytest.raises(SessionError, match="render concurrently"):
                await second._register_active()
        finally:
            await first._unregister_active()
        return first, second

    first, _ = asyncio.run(create())

    async def restore():
        return await Session.restore(first.id, session_root=tmp_path)

    with first._session_file_lock():
        with pytest.raises(SessionLockedError):
            asyncio.run(restore())
