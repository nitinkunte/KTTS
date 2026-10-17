import asyncio

import numpy as np
import pytest

from ktts.audio.audio import AudioNormalizer
from ktts.structures.schemas import NormalizationOptions
from ktts.textproc.normalization import normalize_text
from ktts.textproc.text_processor import get_sentence_info, smart_split


def test_english_normalizer_expands_numbers() -> None:
    assert normalize_text("I have 10 apples.", NormalizationOptions()) == (
        "I have ten apples."
    )


def test_text_processor_preserves_sentence_and_pause_chunks() -> None:
    async def collect_chunks():
        return [
            chunk
            async for chunk in smart_split(
                "Hello world. [pause:0.1s] Bye.",
                normalization_options=NormalizationOptions(),
            )
        ]

    chunks = asyncio.run(collect_chunks())

    assert [chunk[0] for chunk in chunks] == ["Hello world.", "", "Bye."]
    assert chunks[1][1:] == ([], 0.1)


def test_sentence_tokenization_uses_requested_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def process_text_chunk(
        text: str, language: str = "a", lexicon=None
    ) -> list[int]:
        calls.append((text, language, lexicon))
        return [1]

    monkeypatch.setattr(
        "ktts.textproc.text_processor.process_text_chunk", process_text_chunk
    )

    assert list(get_sentence_info("Hello.", lang_code="b")) == [
        ("Hello.", [1], 1)
    ]
    assert calls == [("Hello.", "b", None)]


def test_audio_normalizer_clips_and_converts_to_int16() -> None:
    normalized = AudioNormalizer().normalize(np.array([-1.5, 0.5, 1.5]))

    assert normalized.dtype == np.int16
    assert normalized.tolist() == [-32768, 16383, 32767]
