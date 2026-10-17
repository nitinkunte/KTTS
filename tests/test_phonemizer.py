import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import pytest
from loguru import logger
from phonemizer.backend.espeak.language_switch import RemoveFlags

from ktts.engine import KokoroV1
from ktts.model_manager import ModelManager
from ktts.textproc.phonemizer import (
    EspeakSetupError,
    check_espeak,
    get_english_g2p,
    phonemize,
)
from ktts.textproc.vocabulary import VOCAB, normalize_phonemes, tokenize

FOREIGN_NAME_FIXTURES = (
    "Dvořák",
    "Søren",
    "Nguyễn",
    "Łukasz",
    "Xiaoming",
    "Tchaikovsky",
    "Zorblax",
    "Quenthariel",
    "McKay-Rae",
    "O'Neil",
)


@pytest.mark.parametrize("name", FOREIGN_NAME_FIXTURES)
def test_foreign_and_invented_names_produce_supported_tokens(name: str) -> None:
    phonemes = phonemize(name)

    assert phonemes
    assert not (set(phonemes) - VOCAB.keys())
    assert tokenize(phonemes)


@pytest.mark.parametrize("language", ("a", "b"))
def test_us_and_british_english_g2p_produce_tokens(language: str) -> None:
    phonemes = phonemize("hello world", language)

    assert phonemes
    assert tokenize(phonemes)
    assert not (set(phonemes) - VOCAB.keys())


def test_espeak_fallback_removes_language_switch_markers() -> None:
    assert isinstance(
        get_english_g2p("a")._g2p.fallback.backend._lang_switch,
        RemoveFlags,
    )


def test_raised_schwa_uses_documented_vocabulary_substitution() -> None:
    assert normalize_phonemes("ˈkwɛnθɑɹɹiᵊl") == "ˈkwɛnθɑɹɹiəl"


def test_pronunciation_lexicon_overrides_g2p() -> None:
    assert phonemize(
        "Quenthariel",
        lexicon={"Quenthariel": "kwɛnθɑɹiəl"},
    ) == "kwɛnθɑɹiəl"


def test_unmapped_symbols_are_reported_instead_of_silently_dropped() -> None:
    output = StringIO()
    sink_id = logger.add(output)
    try:
        assert tokenize("hello ☃")
    finally:
        logger.remove(sink_id)

    assert "Unmapped IPA symbols" in output.getvalue()
    assert "☃" in output.getvalue()


def test_invalid_espeak_data_path_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ESPEAK_DATA_PATH", "/path/that/does/not/exist")

    with pytest.raises(EspeakSetupError, match="brew install espeak-ng"):
        check_espeak()


def test_startup_check_exercises_the_espeak_fallback() -> None:
    check_espeak()


def test_model_manager_checks_espeak_before_initializing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("ktts.model_manager.check_espeak", lambda: calls.append("espeak"))
    monkeypatch.setattr(
        "ktts.model_manager.KokoroV1",
        lambda: calls.append("backend"),
    )

    asyncio.run(ModelManager().initialize())

    assert calls == ["espeak", "backend"]


def test_concurrent_phonemization_is_serialized_and_stable() -> None:
    expected = phonemize("Quenthariel")
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(phonemize, ("Quenthariel",) * 32))

    assert outputs == [expected] * 32


def test_kokoro_pipeline_uses_shared_english_g2p(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pipeline:
        def __init__(self, **kwargs: object) -> None:
            self.g2p = None

    monkeypatch.setattr("ktts.engine.KPipeline", Pipeline)
    engine = KokoroV1()
    engine._model = object()

    pipeline = engine._get_pipeline("a")

    assert pipeline.g2p is get_english_g2p("a")
    assert pipeline.g2p("Quenthariel")[0] == phonemize("Quenthariel")
