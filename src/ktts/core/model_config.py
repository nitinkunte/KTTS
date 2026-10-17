"""Configuration for the local Kokoro model file."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_file: str = "v1_0/kokoro-v1_0.pth"


model_config = ModelConfig()
