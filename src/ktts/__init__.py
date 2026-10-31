"""Kokoro text-to-speech CLI and library."""

from typing import TYPE_CHECKING

__version__ = "0.1.0"

if TYPE_CHECKING:
    from .session import Session


def __getattr__(name: str):
    if name == "Session":
        from .session import Session

        return Session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Session", "__version__"]
