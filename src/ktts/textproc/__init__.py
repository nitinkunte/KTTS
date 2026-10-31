"""Text processing pipeline."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .normalization import get_normalizer, normalize_text
    from .phonemizer import phonemize
    from .text_processor import process_text_chunk, smart_split
    from .vocabulary import tokenize


_EXPORT_MODULES = {
    "get_normalizer": ".normalization",
    "normalize_text": ".normalization",
    "phonemize": ".phonemizer",
    "process_text_chunk": ".text_processor",
    "smart_split": ".text_processor",
    "tokenize": ".vocabulary",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "get_normalizer",
    "normalize_text",
    "phonemize",
    "tokenize",
    "process_text_chunk",
    "smart_split",
]
