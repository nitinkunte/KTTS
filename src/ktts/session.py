"""Persistent, cooperative pause/resume sessions."""

import asyncio
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import uuid
import wave
from collections.abc import AsyncIterator, Callable
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from loguru import logger

from .audio.audio import AudioNormalizer, AudioService
from .base import AudioChunk
from .core import paths
from .core.config import settings
from .engine import PlannedChunk, plan
from .model_manager import ModelManager, get_manager
from .structures.schemas import NormalizationOptions

PLAN_VERSION = 1
SESSION_FORMAT_VERSION = 1
_SESSION_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
_ACTIVE_SESSION_IDS: set[str] = set()
_ACTIVE_SESSION_LOCK = threading.Lock()


class SessionError(RuntimeError):
    """Base exception for session lifecycle failures."""


class SessionLockedError(SessionError):
    """Raised when another process already owns a session."""


class SessionPlanMismatchError(SessionError):
    """Raised when the saved deterministic chunk plan has changed."""


class SessionExpiredError(SessionError):
    """Raised when a session has exceeded its configured resume lifetime."""


@contextmanager
def _directory_lock(session_dir: Path):
    lock_path = session_dir / "session.lock"
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SessionLockedError(
                f"Session in {session_dir} is already active in another process"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _serialize_chunk(chunk: PlannedChunk) -> dict[str, object]:
    return {
        "text": chunk.text,
        "tokens": list(chunk.tokens),
        "phonemes": chunk.phonemes,
        "pause_duration_s": chunk.pause_duration_s,
    }


def _deserialize_chunk(data: dict[str, object]) -> PlannedChunk:
    tokens = data["tokens"]
    if not isinstance(tokens, list):
        raise SessionError("Saved chunk tokens must be a list")
    return PlannedChunk(
        text=str(data["text"]),
        tokens=tuple(int(token) for token in tokens),
        phonemes=str(data["phonemes"]),
        pause_duration_s=(
            None
            if data["pause_duration_s"] is None
            else float(data["pause_duration_s"])
        ),
    )


def _hash_plan(
    text: str,
    lang_code: str,
    options: NormalizationOptions,
    chunks: list[PlannedChunk],
    lexicon: Optional[dict[str, str]] = None,
    strict: bool = False,
) -> str:
    payload = {
        "plan_version": PLAN_VERSION,
        "packages": _package_versions(),
        "text": text,
        "lang_code": lang_code,
        "options": asdict(options),
        "lexicon": lexicon or {},
        "strict": strict,
        "chunks": [_serialize_chunk(chunk) for chunk in chunks],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _package_versions() -> dict[str, str]:
    return {
        "kokoro": version("kokoro"),
        "misaki": version("misaki"),
        "phonemizer-fork": version("phonemizer-fork"),
    }


def _remaining_text(chunks: list[PlannedChunk]) -> str:
    parts = []
    for chunk in chunks:
        if chunk.pause_duration_s is not None:
            parts.append(f"[pause:{chunk.pause_duration_s:g}s]")
        elif chunk.text:
            parts.append(chunk.text)
    return " ".join(parts)


def list_saved_sessions(
    session_root: Optional[Path | str] = None,
) -> list[dict[str, object]]:
    root = Path(session_root or settings.sessions_dir).expanduser()
    if not root.exists():
        return []
    records = []
    for session_dir in root.iterdir():
        if not session_dir.is_dir() or not _SESSION_ID_PATTERN.fullmatch(
            session_dir.name
        ):
            continue
        with (session_dir / "session.json").open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        total = len(metadata["chunks"])
        cursor = int(metadata["cursor"])
        records.append(
            {
                "id": session_dir.name,
                "state": str(metadata["state"]),
                "cursor": cursor,
                "total": total,
                "updated_at": str(metadata["updated_at"]),
                "output_path": str(metadata.get("output_path", "")),
            }
        )
    return sorted(records, key=lambda record: str(record["updated_at"]), reverse=True)


def latest_unfinished_session_id(
    session_root: Optional[Path | str] = None,
) -> str:
    ttl = timedelta(days=settings.session_ttl_days)
    now = datetime.now(timezone.utc)
    for record in list_saved_sessions(session_root):
        if record["state"] == "done":
            continue
        updated_at = datetime.fromisoformat(str(record["updated_at"]))
        if now - updated_at <= ttl:
            return str(record["id"])
    raise SessionError(
        "No unfinished saved session was found. Run `ktts --sessions` to inspect saved sessions."
    )


class _SessionAudioWriter:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir

    def chunk_path(self, index: int) -> Path:
        return self.session_dir / f"chunk-{index:06d}.wav"

    def write_chunk(self, index: int, audio: np.ndarray) -> None:
        destination = self.chunk_path(index)
        temporary = self._temporary_path(destination)
        try:
            with wave.open(str(temporary), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                output.writeframes(np.asarray(audio, dtype=np.int16).tobytes())
            with temporary.open("rb+") as audio_file:
                os.fsync(audio_file.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def stitch(self, chunk_count: int, destination: Optional[Path] = None) -> Path:
        output_path = destination or self.session_dir / "output.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._temporary_path(output_path)
        try:
            with wave.open(str(temporary), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                for index in range(chunk_count):
                    chunk_path = self.chunk_path(index)
                    if not chunk_path.is_file():
                        raise SessionError(
                            f"Cannot export session: missing rendered chunk {index}"
                        )
                    with wave.open(str(chunk_path), "rb") as chunk_file:
                        if (
                            chunk_file.getnchannels() != 1
                            or chunk_file.getsampwidth() != 2
                            or chunk_file.getframerate() != 24000
                        ):
                            raise SessionError(
                                f"Rendered chunk {index} has unexpected WAV format"
                            )
                        frame_count = chunk_file.getnframes()
                        audio = chunk_file.readframes(frame_count)
                        if len(audio) != frame_count * 2:
                            raise SessionError(
                                f"Rendered chunk {index} is truncated"
                            )
                        output.writeframes(audio)
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return output_path

    @staticmethod
    def _temporary_path(destination: Path) -> Path:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            return Path(temporary.name)


class Session:
    """One durable synthesis plan with cooperative pause and resume controls."""

    def __init__(
        self,
        session_id: str,
        text: str,
        voice: str,
        voice_path: str,
        speed: float,
        lang_code: str,
        normalization_options: NormalizationOptions,
        chunks: list[PlannedChunk],
        session_root: Path,
        manager: Optional[ModelManager] = None,
        renderer: Optional[Callable[[PlannedChunk], AsyncIterator[AudioChunk]]] = None,
        cursor: int = 0,
        state: str = "idle",
        output_path: Optional[Path] = None,
        lexicon: Optional[dict[str, str]] = None,
        strict: bool = False,
        plan_mode: str = "full",
        plan_prefix_count: int = 0,
        plan_tail_text: str = "",
    ) -> None:
        self.id = session_id
        self.text = text
        self.voice = voice
        self.voice_path = voice_path
        self.speed = speed
        self.lang_code = lang_code
        self.normalization_options = normalization_options
        self.chunks = chunks
        self.cursor = cursor
        self.state = state
        self.plan_mode = plan_mode
        self.plan_prefix_count = plan_prefix_count
        self.plan_tail_text = plan_tail_text
        self.session_root = session_root
        self.session_dir = session_root / session_id
        self.output_path = output_path or self.session_dir / "output.wav"
        self.lexicon = lexicon or {}
        self.strict = strict
        self.normalizer = AudioNormalizer()
        self.writer = _SessionAudioWriter(self.session_dir)
        self._manager = manager
        self._renderer = renderer
        self._run_gate = asyncio.Event()
        if state != "paused":
            self._run_gate.set()
        self._stream_lock = asyncio.Lock()
        self._stop_requested = False
        self._stream_active = False
        self.plan_hash = _hash_plan(
            text, lang_code, normalization_options, chunks, self.lexicon, self.strict
        )

    @classmethod
    async def create(
        cls,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        normalization_options: Optional[NormalizationOptions] = None,
        session_root: Optional[Path | str] = None,
        manager: Optional[ModelManager] = None,
        renderer: Optional[Callable[[PlannedChunk], AsyncIterator[AudioChunk]]] = None,
        output_path: Optional[Path | str] = None,
        lexicon: Optional[dict[str, str]] = None,
        strict: bool = False,
    ) -> "Session":
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Speed must be a finite number greater than zero")
        selected_voice = voice or settings.default_voice
        voice_path = await paths.get_voice_path(selected_voice)
        selected_language = lang_code or Path(selected_voice).name[0].lower()
        if selected_language not in {"a", "b"}:
            raise ValueError(f"Unsupported English language code: {selected_language}")
        options = normalization_options or NormalizationOptions()
        chunks = await plan(
            text,
            selected_language,
            options,
            lexicon=lexicon,
            strict=strict,
        )
        root = Path(session_root or settings.sessions_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex
        session = cls(
            session_id=session_id,
            text=text,
            voice=selected_voice,
            voice_path=voice_path,
            speed=speed,
            lang_code=selected_language,
            normalization_options=options,
            chunks=chunks,
            session_root=root,
            manager=manager,
            renderer=renderer,
            output_path=(
                Path(output_path).expanduser().resolve() if output_path else None
            ),
            lexicon=lexicon,
            strict=strict,
        )
        session.session_dir.mkdir(mode=0o700)
        session._save_metadata()
        return session

    @classmethod
    async def restore(
        cls,
        session_id: str,
        session_root: Optional[Path | str] = None,
        on_plan_mismatch: Literal["raise", "restart", "replan"] = "raise",
        output_path: Optional[Path | str] = None,
        manager: Optional[ModelManager] = None,
        renderer: Optional[Callable[[PlannedChunk], AsyncIterator[AudioChunk]]] = None,
    ) -> "Session":
        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError("Session ID must be a 32-character lowercase hex string")
        root = Path(session_root or settings.sessions_dir).expanduser().resolve()
        session_dir = (root / session_id).resolve()
        try:
            session_dir.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "Session path escaped the configured session directory"
            ) from error
        with _directory_lock(session_dir):
            return await cls._restore_locked(
                session_id,
                session_root=root,
                on_plan_mismatch=on_plan_mismatch,
                output_path=output_path,
                manager=manager,
                renderer=renderer,
            )

    @classmethod
    async def _restore_locked(
        cls,
        session_id: str,
        session_root: Optional[Path | str] = None,
        on_plan_mismatch: Literal["raise", "restart", "replan"] = "raise",
        output_path: Optional[Path | str] = None,
        manager: Optional[ModelManager] = None,
        renderer: Optional[Callable[[PlannedChunk], AsyncIterator[AudioChunk]]] = None,
    ) -> "Session":
        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError("Session ID must be a 32-character lowercase hex string")
        if on_plan_mismatch not in {"raise", "restart", "replan"}:
            raise ValueError(f"Unsupported plan-mismatch action: {on_plan_mismatch}")

        root = Path(session_root or settings.sessions_dir).expanduser().resolve()
        session_dir = (root / session_id).resolve()
        try:
            session_dir.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "Session path escaped the configured session directory"
            ) from error
        metadata_path = session_dir / "session.json"
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.get("format_version") != SESSION_FORMAT_VERSION:
            raise SessionError(
                f"Unsupported session format version: {metadata.get('format_version')!r}"
            )

        updated_at = datetime.fromisoformat(metadata["updated_at"])
        if settings.session_ttl_days < 1:
            raise SessionError("SESSION_TTL_DAYS must be at least 1")
        ttl = timedelta(days=settings.session_ttl_days)
        if datetime.now(timezone.utc) - updated_at > ttl:
            raise SessionExpiredError(
                "This session exceeded its resume lifetime and remains on disk. "
                "Create a new session or remove the old session explicitly."
            )

        options = NormalizationOptions(**metadata["normalization_options"])
        text = str(metadata["text"])
        lang_code = str(metadata["lang_code"])
        lexicon = {str(key): str(value) for key, value in metadata.get("lexicon", {}).items()}
        strict = bool(metadata.get("strict", False))
        saved_output = output_path or metadata.get("output_path")
        resolved_output = (
            Path(saved_output).expanduser().resolve() if saved_output else None
        )
        saved_chunks = [_deserialize_chunk(item) for item in metadata["chunks"]]
        cursor = max(0, min(int(metadata["cursor"]), len(saved_chunks)))
        mode = str(metadata.get("plan_mode", "full"))
        prefix_count = int(metadata.get("plan_prefix_count", 0))
        tail_text = str(metadata.get("plan_tail_text", ""))

        if mode == "replanned":
            current_tail = (
                await plan(tail_text, lang_code, options, lexicon, strict)
                if tail_text.strip()
                else []
            )
            current_chunks = saved_chunks[:prefix_count] + current_tail
        else:
            current_chunks = await plan(text, lang_code, options, lexicon, strict)

        current_hash = _hash_plan(
            text, lang_code, options, current_chunks, lexicon, strict
        )
        mismatch = current_hash != metadata["plan_hash"]
        if mismatch and on_plan_mismatch == "raise":
            raise SessionPlanMismatchError(
                "The saved chunk plan differs from the current splitter or "
                "phonemizer. Restore with on_plan_mismatch='restart' to start "
                "over, or 'replan' to rebuild from the next saved chunk text."
            )

        voice = str(metadata["voice"])
        voice_path = await paths.get_voice_path(voice)
        if mismatch and on_plan_mismatch == "restart":
            cls._remove_chunk_files(session_dir)
            (session_dir / "output.wav").unlink(missing_ok=True)
            if resolved_output is not None:
                resolved_output.unlink(missing_ok=True)
            current_chunks = await plan(text, lang_code, options, lexicon, strict)
            cursor = 0
            mode, prefix_count, tail_text = "full", 0, ""
        elif mismatch and on_plan_mismatch == "replan":
            cursor = cls._recover_cursor(session_dir, saved_chunks, cursor)
            prefix = saved_chunks[:cursor]
            tail_text = _remaining_text(saved_chunks[cursor:])
            current_tail = (
                await plan(tail_text, lang_code, options, lexicon, strict)
                if tail_text.strip()
                else []
            )
            current_chunks = prefix + current_tail
            mode, prefix_count = "replanned", cursor
            cls._remove_chunk_files(session_dir, from_index=cursor)
            (session_dir / "output.wav").unlink(missing_ok=True)
            if resolved_output is not None:
                resolved_output.unlink(missing_ok=True)
        else:
            cursor = cls._recover_cursor(session_dir, current_chunks, cursor)
            if mode == "full":
                prefix_count, tail_text = 0, ""

        saved_state = metadata.get("state")
        session = cls(
            session_id=session_id,
            text=text,
            voice=voice,
            voice_path=voice_path,
            speed=float(metadata["speed"]),
            lang_code=lang_code,
            normalization_options=options,
            chunks=current_chunks,
            session_root=root,
            manager=manager,
            renderer=renderer,
            cursor=cursor,
            output_path=resolved_output,
            lexicon=lexicon,
            strict=strict,
            state="done" if not mismatch and saved_state == "done" else "paused",
            plan_mode=mode,
            plan_prefix_count=prefix_count,
            plan_tail_text=tail_text,
        )
        session._save_metadata()
        return session

    @property
    def state_path(self) -> Path:
        return self.session_dir / "session.json"

    @property
    def is_paused(self) -> bool:
        return not self._run_gate.is_set()

    def pause(self) -> None:
        if self.state in {"done", "stopped"}:
            return
        if self._stream_active:
            self._run_gate.clear()
            return
        with self._session_file_lock():
            self._run_gate.clear()
            self.state = "paused"
            self._save_metadata()

    def resume(self) -> None:
        if self.state in {"done", "stopped"}:
            raise SessionError(f"Cannot resume a {self.state} session")
        if self._stream_active:
            self._stop_requested = False
            self._run_gate.set()
            return
        with self._session_file_lock():
            self._stop_requested = False
            if self.cursor < len(self.chunks):
                self.output_path.unlink(missing_ok=True)
            self.state = "running"
            self._run_gate.set()
            self._save_metadata()

    def stop(self) -> None:
        if self.state in {"done", "stopped"}:
            return
        self._stop_requested = True
        self._run_gate.set()
        if self._stream_active:
            return
        with self._session_file_lock():
            self.writer.stitch(self.cursor, self.output_path)
            self.state = "stopped"
            self._save_metadata()

    def seek(self, index: int) -> None:
        if self._stream_active or self.state == "running":
            raise SessionError("Cannot seek while a session is running")
        if self.state in {"done", "stopped"}:
            raise SessionError(f"Cannot seek in a {self.state} session")
        if index < 0 or index > len(self.chunks):
            raise ValueError(f"Chunk index must be between 0 and {len(self.chunks)}")
        with self._session_file_lock():
            self._remove_chunk_files(self.session_dir, from_index=index)
            self.output_path.unlink(missing_ok=True)
            self.cursor = index
            self.state = "paused"
            self._save_metadata()

    def export(self, output_path: Optional[Path | str] = None) -> Path:
        if self._stream_active:
            raise SessionError("Cannot export while a session is running")
        with self._session_file_lock():
            return self.writer.stitch(
                self.cursor,
                Path(output_path).expanduser() if output_path is not None else None,
            )

    async def stream(self) -> AsyncIterator[AudioChunk]:
        async with self._stream_lock:
            if self.state in {"done", "stopped"}:
                return
            with self._session_file_lock():
                await self._register_active()
                self._stream_active = True
                completed = False
                try:
                    if self.state == "idle":
                        self.state = "running"
                    self._save_metadata()
                    while self.cursor < len(self.chunks):
                        await self._wait_for_resume()
                        if self._stop_requested:
                            break
                        chunk = self.chunks[self.cursor]
                        rendered = await self.render(chunk)
                        prepared = self._prepare_audio(
                            rendered,
                            chunk,
                            is_last_chunk=self.cursor == len(self.chunks) - 1,
                        )
                        self.writer.write_chunk(self.cursor, prepared.audio)
                        self.cursor += 1
                        if self.cursor == len(self.chunks):
                            self.state = "done"
                            self.writer.stitch(self.cursor, self.output_path)
                        else:
                            self.state = (
                                "paused" if not self._run_gate.is_set() else "running"
                            )
                        self._save_metadata()
                        yield prepared

                    if self._stop_requested:
                        self.state = "stopped"
                        self.writer.stitch(self.cursor, self.output_path)
                    elif self.state != "done" and self.cursor == len(self.chunks):
                        self.state = "done"
                        self.writer.stitch(self.cursor, self.output_path)
                    self._save_metadata()
                    completed = True
                finally:
                    if not completed and self.state not in {"done", "stopped"}:
                        if self._stop_requested:
                            self.state = "stopped"
                            self.writer.stitch(self.cursor, self.output_path)
                        elif settings.pause_on_disconnect:
                            self._run_gate.clear()
                            self.state = "paused"
                        else:
                            self._stop_requested = True
                            self.state = "stopped"
                            self.writer.stitch(self.cursor, self.output_path)
                        self._save_metadata()
                    self._stream_active = False
                    await self._unregister_active()

    async def render(self, chunk: PlannedChunk) -> AudioChunk:
        if chunk.pause_duration_s is not None:
            samples = round(chunk.pause_duration_s * self.normalizer.sample_rate)
            return AudioChunk(np.zeros(samples, dtype=np.int16))

        audio_parts = []
        if self._renderer is not None:
            async for rendered in self._renderer(chunk):
                audio_parts.append(rendered.audio)
                self._check_pause_between_subchunks()
                await asyncio.sleep(0)
        else:
            manager = self._manager or await get_manager()
            async for rendered in manager.generate_from_tokens(
                chunk.phonemes,
                self.voice_path,
                self.speed,
                self.lang_code,
            ):
                audio_parts.append(rendered.audio)
                self._check_pause_between_subchunks()
                await asyncio.sleep(0)

        if not audio_parts:
            raise SessionError(f"Model generated no audio for chunk: {chunk.text!r}")
        return AudioChunk(np.concatenate(audio_parts))

    def _check_pause_between_subchunks(self) -> None:
        if not self._run_gate.is_set():
            logger.debug("Pause requested; finishing the current planned chunk")

    def _prepare_audio(
        self, rendered: AudioChunk, chunk: PlannedChunk, is_last_chunk: bool
    ) -> AudioChunk:
        if chunk.pause_duration_s is not None:
            rendered.audio = self.normalizer.normalize(rendered.audio)
            return rendered
        rendered = AudioService.trim_audio(
            rendered,
            chunk_text=chunk.text,
            speed=self.speed,
            is_last_chunk=is_last_chunk,
            normalizer=self.normalizer,
        )
        rendered.audio = self.normalizer.normalize(rendered.audio)
        return rendered

    async def _wait_for_resume(self) -> None:
        while not self._run_gate.is_set() and not self._stop_requested:
            self.state = "paused"
            self._save_metadata()
            await self._run_gate.wait()
        if not self._stop_requested:
            self.state = "running"
            self._save_metadata()

    async def _register_active(self) -> None:
        with _ACTIVE_SESSION_LOCK:
            active_limit = settings.max_concurrent_sessions
            if active_limit < 1:
                raise SessionError("MAX_CONCURRENT_SESSIONS must be at least 1")
            if len(_ACTIVE_SESSION_IDS) >= active_limit:
                raise SessionError(
                    f"At most {active_limit} sessions may render concurrently"
                )
            _ACTIVE_SESSION_IDS.add(self.id)

    async def _unregister_active(self) -> None:
        with _ACTIVE_SESSION_LOCK:
            _ACTIVE_SESSION_IDS.discard(self.id)

    @contextmanager
    def _session_file_lock(self):
        with _directory_lock(self.session_dir):
            yield

    def _save_metadata(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": SESSION_FORMAT_VERSION,
            "plan_version": PLAN_VERSION,
            "package_versions": _package_versions(),
            "session_id": self.id,
            "text": self.text,
            "voice": self.voice,
            "speed": self.speed,
            "output_path": str(self.output_path),
            "lang_code": self.lang_code,
            "normalization_options": asdict(self.normalization_options),
            "lexicon": self.lexicon,
            "strict": self.strict,
            "chunks": [_serialize_chunk(chunk) for chunk in self.chunks],
            "cursor": self.cursor,
            "state": self.state,
            "plan_hash": self.plan_hash,
            "plan_mode": self.plan_mode,
            "plan_prefix_count": self.plan_prefix_count,
            "plan_tail_text": self.plan_tail_text,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = self.session_dir / f".session-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as output:
                json.dump(metadata, output, ensure_ascii=False, sort_keys=True)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.state_path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _recover_cursor(
        cls, session_dir: Path, chunks: list[PlannedChunk], cursor: int
    ) -> int:
        writer = _SessionAudioWriter(session_dir)
        contiguous = 0
        for index in range(len(chunks)):
            path = writer.chunk_path(index)
            if not path.is_file():
                break
            try:
                with wave.open(str(path), "rb") as chunk_file:
                    valid_format = (
                        chunk_file.getnchannels() == 1
                        and chunk_file.getsampwidth() == 2
                        and chunk_file.getframerate() == 24000
                    )
                    expected_size = 44 + chunk_file.getnframes() * 2
                    valid_data = path.stat().st_size == expected_size
            except (EOFError, wave.Error) as error:
                logger.warning(
                    f"Discarding incomplete rendered chunk {index}: {error}"
                )
                cls._remove_chunk_files(session_dir, from_index=index)
                break
            if not valid_format or not valid_data:
                logger.warning(
                    f"Discarding incomplete or invalid rendered chunk {index}"
                )
                cls._remove_chunk_files(session_dir, from_index=index)
                break
            contiguous += 1
        if contiguous < cursor:
            logger.warning(
                f"Session cursor was {cursor}, but only {contiguous} complete "
                "chunk files exist; resuming from the first missing chunk"
            )
        if contiguous > cursor:
            logger.info(
                f"Recovered {contiguous - cursor} chunk file(s) written before "
                "the last session-state update"
            )
        cls._remove_chunk_files(session_dir, from_index=contiguous)
        return contiguous

    @staticmethod
    def _remove_chunk_files(session_dir: Path, from_index: int = 0) -> None:
        for path in session_dir.glob("chunk-*.wav"):
            match = re.fullmatch(r"chunk-(\d{6})\.wav", path.name)
            if match and int(match.group(1)) >= from_index:
                path.unlink()
        for path in session_dir.glob(".chunk-*.wav.*.tmp"):
            match = re.fullmatch(r"\.chunk-(\d{6})\.wav\..+\.tmp", path.name)
            if match and int(match.group(1)) >= from_index:
                path.unlink()
