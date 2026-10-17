import os
import re
import threading
from pathlib import Path

from loguru import logger
from misaki import en, espeak
from misaki.token import MToken
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper

from .vocabulary import normalize_phonemes

_phonemizers: dict[str, "KokoroEnglishG2P"] = {}
_phonemizer_lock = threading.RLock()
_data_path_lock = threading.Lock()


class EspeakSetupError(RuntimeError):
    """Raised when the eSpeak NG engine or its data cannot be initialized."""


class KokoroEspeakFallback(espeak.EspeakFallback):
    def __init__(self, british: bool):
        self.british = british
        self.version = None
        self.backend = EspeakBackend(
            language=f"en-{'gb' if british else 'us'}",
            preserve_punctuation=True,
            with_stress=True,
            tie="^",
            language_switch="remove-flags",
        )


def _configured_data_path() -> str | None:
    for name in (
        "ESPEAK_DATA_PATH",
        "PHONEMIZER_ESPEAK_DATA",
        "PHONEMIZER_ESPEAK_DATA_PATH",
    ):
        path = os.environ.get(name)
        if path:
            data_path = Path(path).expanduser()
            if not data_path.is_dir():
                raise EspeakSetupError(
                    f"{name} points to a missing eSpeak NG data directory: "
                    f"{data_path}. Install it with `brew install espeak-ng`, or "
                    "set the variable to `$(brew --prefix)/share/espeak-ng-data`."
                )
            return str(data_path)
    return None


def _configure_espeak() -> None:
    configured_path = _configured_data_path()
    if configured_path is not None:
        with _data_path_lock:
            EspeakWrapper.set_data_path(configured_path)


class KokoroEnglishG2P:
    """English Misaki G2P configured like Kokoro's KPipeline."""

    def __init__(self, language: str):
        if language not in {"a", "b"}:
            raise ValueError(f"Unsupported language code: {language}")
        _configure_espeak()
        british = language == "b"
        fallback = KokoroEspeakFallback(british=british)
        self._g2p = en.G2P(
            trf=False,
            british=british,
            fallback=fallback,
            unk="",
        )

    def __call__(self, text: str, preprocess: bool = True) -> tuple[str, list[MToken]]:
        with _phonemizer_lock:
            phonemes, tokens = self._g2p(text, preprocess=preprocess)
            phonemes = normalize_phonemes(phonemes)
            for token in tokens:
                if token.phonemes is not None:
                    token.phonemes = normalize_phonemes(token.phonemes, warn=False)
            return phonemes, tokens


def get_english_g2p(language: str = "a") -> KokoroEnglishG2P:
    if language not in {"a", "b"}:
        raise ValueError(f"Unsupported language code: {language}")
    _configure_espeak()
    with _phonemizer_lock:
        if language not in _phonemizers:
            _phonemizers[language] = KokoroEnglishG2P(language)
        return _phonemizers[language]


def phonemize(
    text: str, language: str = "a", lexicon: dict[str, str] | None = None
) -> str:
    text = text.strip()
    if not lexicon:
        return get_english_g2p(language)(text)[0].strip()

    replacements = {word.casefold(): phonemes for word, phonemes in lexicon.items()}
    alternatives = "|".join(
        re.escape(word)
        for word in sorted(lexicon, key=len, reverse=True)
    )
    pattern = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)
    pieces = []
    position = 0
    for match in pattern.finditer(text):
        pieces.append(get_english_g2p(language)(text[position : match.start()])[0])
        pieces.append(replacements[match.group().casefold()])
        position = match.end()
    if not pieces:
        return get_english_g2p(language)(text)[0].strip()
    pieces.append(get_english_g2p(language)(text[position:])[0])
    return normalize_phonemes(
        " ".join(piece.strip() for piece in pieces if piece.strip())
    )


def check_espeak() -> None:
    """Exercise the English OOV fallback so missing eSpeak data fails early."""
    try:
        phonemes = phonemize("Kokoro startup check Zorblax")
    except (EspeakSetupError, OSError, RuntimeError) as error:
        raise EspeakSetupError(
            "eSpeak NG is required for English out-of-vocabulary words. Install "
            "it with `brew install espeak-ng`; if its data is not found, set "
            "ESPEAK_DATA_PATH or PHONEMIZER_ESPEAK_DATA_PATH to "
            "`$(brew --prefix)/share/espeak-ng-data`."
        ) from error
    if not phonemes:
        raise EspeakSetupError(
            "eSpeak NG startup check produced no phonemes. Install it with "
            "`brew install espeak-ng` and verify the eSpeak data directory."
        )
