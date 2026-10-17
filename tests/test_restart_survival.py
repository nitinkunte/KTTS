import asyncio
import os
import select
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from ktts.base import AudioChunk
from ktts.engine import PlannedChunk
from ktts.session import Session

_CHILD_PROGRAM = textwrap.dedent(
    """
    import asyncio
    import sys
    import numpy as np
    from unittest.mock import patch
    from ktts.base import AudioChunk
    from ktts.engine import PlannedChunk
    from ktts.session import Session

    async def build_plan(*args, **kwargs):
        return [
            PlannedChunk(text="First.", tokens=(1,), phonemes="fˈɜːst."),
            PlannedChunk(text="Second.", tokens=(1,), phonemes="sˈɛkənd."),
        ]

    async def voice_path(_):
        return "/fake/af_heart.pt"

    async def render(chunk):
        if chunk.text == "Second." and sys.argv[2] == "block":
            print("SECOND_CHUNK_STARTED", flush=True)
            await asyncio.sleep(120)
        value = 0.2 if chunk.text == "First." else 0.4
        yield AudioChunk(np.full(6000, value, dtype=np.float32))

    async def main():
        with patch("ktts.session.plan", build_plan), patch(
            "ktts.session.paths.get_voice_path", voice_path
        ):
            session = await Session.restore(
                sys.argv[1],
                session_root=sys.argv[3],
                renderer=render,
            )
            session.resume()
            async for _ in session.stream():
                print(f"CHUNK_SAVED {session.cursor}", flush=True)
            print(f"STATE {session.state}", flush=True)

    asyncio.run(main())
    """
)


def _install_parent_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def build_plan(*args, **kwargs):
        return [
            PlannedChunk(text="First.", tokens=(1,), phonemes="fˈɜːst."),
            PlannedChunk(text="Second.", tokens=(1,), phonemes="sˈɛkənd."),
        ]

    async def voice_path(_: str) -> str:
        return "/fake/af_heart.pt"

    monkeypatch.setattr("ktts.session.plan", build_plan)
    monkeypatch.setattr("ktts.session.paths.get_voice_path", voice_path)


def _start_child(session: Session, mode: str) -> subprocess.Popen[bytes]:
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_path = str(repository_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, environment.get("PYTHONPATH", "")) if part
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CHILD_PROGRAM,
            session.id,
            mode,
            str(session.session_root),
        ],
        cwd=repository_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def _wait_for_line(process: subprocess.Popen[bytes], expected: str) -> list[str]:
    assert process.stdout is not None
    output = bytearray()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        ready, _, _ = select.select([process.stdout], [], [], 0.2)
        if ready:
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            output.extend(chunk)
            if expected.encode() in output:
                return output.decode(errors="replace").splitlines()
        if process.poll() is not None:
            break
    process.kill()
    stdout, stderr = process.communicate(timeout=30)
    raise AssertionError(
        f"Child did not print {expected!r}; "
        f"stdout={(output + stdout).decode(errors='replace')!r}; "
        f"stderr={stderr.decode(errors='replace') if stderr else ''!r}"
    )


def _new_session(monkeypatch: pytest.MonkeyPatch, root: Path) -> Session:
    _install_parent_fakes(monkeypatch)

    async def create():
        return await Session.create(
            "First. Second.",
            session_root=root,
            renderer=_parent_renderer(),
        )

    return asyncio.run(create())


def _parent_renderer():
    async def render(chunk: PlannedChunk):
        value = 0.2 if chunk.text == "First." else 0.4
        yield AudioChunk(np.full(6000, value, dtype=np.float32))

    return render


def test_sigint_mid_chunk_persists_and_resumes_to_uninterrupted_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interrupted = _new_session(monkeypatch, tmp_path / "interrupted")
    process = _start_child(interrupted, "block")
    _wait_for_line(process, "SECOND_CHUNK_STARTED")
    time.sleep(0.2)
    process.send_signal(signal.SIGINT)
    process.communicate(timeout=30)
    assert process.returncode in {130, -signal.SIGINT}

    restored = asyncio.run(
        Session.restore(
            interrupted.id,
            session_root=interrupted.session_root,
            renderer=_parent_renderer(),
        )
    )
    assert restored.state == "paused"
    assert restored.cursor == 1
    restored.resume()
    asyncio.run(_consume(restored))

    baseline = _new_session(monkeypatch, tmp_path / "baseline")
    asyncio.run(_consume(baseline))
    assert restored.output_path.read_bytes() == baseline.output_path.read_bytes()


def test_sigkill_mid_chunk_discards_only_incomplete_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interrupted = _new_session(monkeypatch, tmp_path / "killed")
    process = _start_child(interrupted, "block")
    _wait_for_line(process, "SECOND_CHUNK_STARTED")
    process.send_signal(signal.SIGKILL)
    process.communicate(timeout=30)
    assert process.returncode == -signal.SIGKILL
    assert interrupted.writer.chunk_path(0).is_file()
    assert not interrupted.writer.chunk_path(1).exists()

    restored = asyncio.run(
        Session.restore(
            interrupted.id,
            session_root=interrupted.session_root,
            renderer=_parent_renderer(),
        )
    )
    assert restored.cursor == 1
    restored.resume()
    asyncio.run(_consume(restored))
    assert restored.cursor == 2
    assert restored.writer.chunk_path(0).is_file()
    assert restored.writer.chunk_path(1).is_file()


async def _consume(session: Session) -> None:
    async for _ in session.stream():
        pass
