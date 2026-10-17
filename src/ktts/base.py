"""Base interface for Kokoro inference."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import numpy as np
import torch

from .structures.schemas import WordTimestamp


@dataclass
class AudioChunk:
    """Class for audio chunks returned by model backends"""

    audio: np.ndarray
    word_timestamps: Optional[list[WordTimestamp]] = None
    output: Optional[bytes | np.ndarray] = b""


class ModelBackend(ABC):
    """Abstract base class for model inference backend."""

    @abstractmethod
    async def load_model(self, path: str) -> None:
        """Load model from path.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    async def generate(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate audio from text.

        Args:
            text: Input text to synthesize
            voice: Path to the selected voice file
            speed: Speed multiplier

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Get device model is running on.

        Returns:
            Device string
        """
        pass


class BaseModelBackend(ModelBackend):
    """Base implementation of model backend."""

    def __init__(self):
        """Initialize base backend."""
        self._model: Optional[torch.nn.Module] = None
        self._device: str = "cpu"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._model is not None:
            del self._model
            self._model = None
