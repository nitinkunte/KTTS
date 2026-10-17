"""CPU inference backend for Kokoro V1."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncGenerator, Optional

import numpy as np
import torch
from kokoro import KModel, KPipeline
from loguru import logger

from .base import AudioChunk, BaseModelBackend
from .core import paths
from .core.config import settings
from .structures.schemas import NormalizationOptions, WordTimestamp
from .textproc.phonemizer import get_english_g2p, phonemize
from .textproc.text_processor import check_pause_budget, check_speakable, smart_split

if TYPE_CHECKING:
    from .session import Session


@dataclass(frozen=True)
class PlannedChunk:
    text: str
    tokens: tuple[int, ...]
    phonemes: str
    pause_duration_s: Optional[float] = None


async def plan(
    text: str,
    lang_code: str = "a",
    normalization_options: Optional[NormalizationOptions] = None,
    lexicon: Optional[dict[str, str]] = None,
    strict: bool = False,
) -> list[PlannedChunk]:
    options = normalization_options or NormalizationOptions()
    check_speakable(text, options)
    check_pause_budget(text)
    planned: list[PlannedChunk] = []
    async for chunk_text, token_ids, pause_duration in smart_split(
        text,
        lang_code=lang_code,
        normalization_options=options,
        lexicon=lexicon,
    ):
        phonemes = (
            phonemize(chunk_text, lang_code, lexicon)
            if pause_duration is None
            else ""
        )
        if strict:
            from .textproc.vocabulary import VOCAB, normalize_phonemes

            normalized = normalize_phonemes(phonemes, warn=False)
            unmapped = sorted(set(normalized) - VOCAB.keys())
            if unmapped:
                raise ValueError(
                    f"Strict phoneme validation failed for {chunk_text!r}; "
                    f"unmapped symbols: {unmapped!r}"
                )
        planned.append(
            PlannedChunk(
                text=chunk_text,
                tokens=tuple(token_ids),
                phonemes=phonemes,
                pause_duration_s=pause_duration,
            )
        )
    return planned


async def start(
    text: str,
    voice: Optional[str] = None,
    speed: float = 0.9,
    lang_code: Optional[str] = None,
    output_path: Optional[str] = None,
    lexicon: Optional[dict[str, str]] = None,
    strict: bool = False,
) -> "Session":
    """Create a durable synthesis session for the public library API."""
    from .session import Session

    return await Session.create(
        text,
        voice=voice,
        speed=speed,
        lang_code=lang_code,
        output_path=output_path,
        lexicon=lexicon,
        strict=strict,
    )


class KokoroV1(BaseModelBackend):
    def __init__(self) -> None:
        super().__init__()
        self._device = "cpu"
        self._model: Optional[KModel] = None
        self._pipelines: dict[str, KPipeline] = {}

    async def load_model(self, path: str) -> None:
        try:
            model_path = await paths.get_model_path(path)
            config_path = os.path.join(os.path.dirname(model_path), "config.json")
            if not os.path.exists(config_path):
                raise RuntimeError(f"Config file not found: {config_path}")

            logger.info(f"Loading Kokoro model on CPU from {model_path}")
            self._model = KModel(
                repo_id=settings.model_repo_id,
                config=config_path,
                model=model_path,
            ).eval().cpu()
        except FileNotFoundError:
            raise
        except Exception as error:
            raise RuntimeError(f"Failed to load Kokoro model: {error}") from error

    def _get_pipeline(self, lang_code: str) -> KPipeline:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        if lang_code not in {"a", "b"}:
            raise ValueError(f"Unsupported English language code: {lang_code}")
        if lang_code not in self._pipelines:
            pipeline = KPipeline(
                lang_code=lang_code,
                repo_id=settings.model_repo_id,
                model=self._model,
                device="cpu",
            )
            pipeline.g2p = get_english_g2p(lang_code)
            self._pipelines[lang_code] = pipeline
        return self._pipelines[lang_code]

    async def generate_from_tokens(
        self,
        tokens: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> AsyncGenerator[np.ndarray, None]:
        pipeline = self._get_pipeline(lang_code)
        try:
            for result in pipeline.generate_from_tokens(
                tokens=tokens,
                voice=voice,
                speed=speed,
                model=self._model,
            ):
                if result.audio is not None:
                    yield result.audio.numpy()
                else:
                    logger.warning("Kokoro returned a token result without audio")
        except Exception as error:
            raise RuntimeError(f"Generation from tokens failed: {error}") from error

    async def generate(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        return_timestamps: bool = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        selected_language = lang_code or os.path.basename(voice)[0].lower()
        pipeline = self._get_pipeline(selected_language)
        try:
            for result in pipeline(
                text,
                voice=voice,
                speed=speed,
                model=self._model,
            ):
                if result.audio is None:
                    logger.warning("Kokoro returned a result without audio")
                    continue

                timestamps = None
                if return_timestamps and result.tokens:
                    timestamps = [
                        WordTimestamp(
                            word=token.text.strip(),
                            start_time=float(token.start_ts),
                            end_time=float(token.end_ts),
                        )
                        for token in result.tokens
                        if token.text
                        and token.text.strip()
                        and token.start_ts is not None
                        and token.end_ts is not None
                    ]
                yield AudioChunk(result.audio.numpy(), word_timestamps=timestamps)
        except Exception as error:
            raise RuntimeError(f"Generation failed: {error}") from error

    def unload(self) -> None:
        self._model = None
        self._pipelines.clear()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device
