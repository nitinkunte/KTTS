"""Lifecycle management for the local Kokoro model."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from loguru import logger

from .base import AudioChunk
from .core.config import settings
from .core.model_config import ModelConfig, model_config
from .engine import KokoroV1
from .textproc.phonemizer import check_espeak


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self._config = config or model_config
        self._backend: Optional[KokoroV1] = None
        self._lock = asyncio.Lock()
        self._active_requests = 0
        self._last_used_at: Optional[float] = None
        self._idle_unload_task: Optional[asyncio.Task[None]] = None

    async def initialize(self) -> None:
        check_espeak()
        self._backend = KokoroV1()

    async def ensure_backend(self) -> None:
        if self._backend is not None:
            return
        async with self._lock:
            if self._backend is None:
                self._cancel_idle_unload_timer()
                await self.initialize()
                await self.load_model(self._config.model_file)

    async def load_model(self, path: str) -> None:
        if self._backend is None:
            raise RuntimeError("Backend not initialized")
        await self._backend.load_model(path)
        self._last_used_at = time.monotonic()
        self._schedule_idle_unload_timer()

    def _auto_unload_timeout(self) -> float:
        return max(0.0, float(settings.model_auto_unload_timeout_seconds))

    def _cancel_idle_unload_timer(self) -> None:
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        if (
            self._idle_unload_task is not None
            and not self._idle_unload_task.done()
            and self._idle_unload_task is not current_task
        ):
            self._idle_unload_task.cancel()
        self._idle_unload_task = None

    def _schedule_idle_unload_timer(self) -> None:
        self._cancel_idle_unload_timer()
        timeout = self._auto_unload_timeout()
        if timeout > 0 and self._backend is not None and self._active_requests == 0:
            self._idle_unload_task = asyncio.create_task(self._idle_unload_after(timeout))

    async def _idle_unload_after(self, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
            async with self._lock:
                if self._active_requests or self._backend is None:
                    return
                last_used_at = self._last_used_at
                if last_used_at is not None:
                    idle_for = time.monotonic() - last_used_at
                    if idle_for < self._auto_unload_timeout():
                        self._schedule_idle_unload_timer()
                        return
                self._unload_backend()
            logger.info(f"Model unloaded after {timeout:g}s of inactivity")
        except asyncio.CancelledError:
            pass

    def _unload_backend(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
            self._last_used_at = time.monotonic()

    async def _begin_request(self) -> None:
        async with self._lock:
            self._active_requests += 1
            self._cancel_idle_unload_timer()

    async def _end_request(self) -> None:
        async with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._last_used_at = time.monotonic()
            self._schedule_idle_unload_timer()

    @asynccontextmanager
    async def hold(self):
        await self._begin_request()
        try:
            yield
        finally:
            await self._end_request()

    async def generate(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        return_timestamps: bool = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        async with self.hold():
            await self.ensure_backend()
            if self._backend is None:
                raise RuntimeError("Model backend was not initialized")
            async for chunk in self._backend.generate(
                text,
                voice,
                speed,
                lang_code,
                return_timestamps,
            ):
                yield chunk

    async def generate_from_tokens(
        self,
        tokens: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> AsyncGenerator[AudioChunk, None]:
        async with self.hold():
            await self.ensure_backend()
            if self._backend is None:
                raise RuntimeError("Model backend was not initialized")
            async for audio in self._backend.generate_from_tokens(
                tokens,
                voice,
                speed,
                lang_code,
            ):
                yield AudioChunk(audio)

    def unload_all(self) -> None:
        self._cancel_idle_unload_timer()
        self._unload_backend()

    async def unload(self) -> None:
        async with self._lock:
            self._cancel_idle_unload_timer()
            self._unload_backend()

    async def reload(self) -> None:
        async with self._lock:
            self._cancel_idle_unload_timer()
            self._unload_backend()
            await self.initialize()
            await self.load_model(self._config.model_file)


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager(config)
    return ModelManager._instance
