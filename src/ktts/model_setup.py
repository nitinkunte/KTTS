"""Download the Kokoro model and voice assets used by KTTS."""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download
from loguru import logger

from .core.config import settings
from .core.model_config import model_config

_VOICE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
MODEL_CONFIG_FILENAME = "config.json"


def _download_asset(
    repo_id: str,
    filename: str,
    destination: Path,
    downloader: Callable[..., str] = hf_hub_download,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    source = Path(downloader(repo_id=repo_id, filename=filename))
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Downloaded asset is missing or empty: {filename}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded asset is empty: {filename}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _validated_voices(voices: list[str]) -> list[str]:
    unique_voices = list(dict.fromkeys(voices))
    if not unique_voices:
        raise ValueError("At least one voice name is required")
    invalid = [voice for voice in unique_voices if not _VOICE_PATTERN.fullmatch(voice)]
    if invalid:
        raise ValueError(
            "Voice names must contain lowercase letters, digits, and underscores: "
            f"{invalid!r}"
        )
    return unique_voices


def download_voices(
    voices: list[str],
    voices_dir: str | Path | None = None,
    downloader: Callable[..., str] = hf_hub_download,
) -> list[Path]:
    """Download individual voice `.pt` files, skipping ones already installed."""
    unique_voices = _validated_voices(voices)
    voice_root = Path(voices_dir or settings.voices_dir).expanduser()
    destinations = []
    for voice in unique_voices:
        destinations.append(
            _download_asset(
                settings.model_repo_id,
                f"voices/{voice}.pt",
                voice_root / f"{voice}.pt",
                downloader,
            )
        )
    logger.info(f"Installed {len(destinations)} voice(s) in {voice_root}")
    return destinations


def setup_model_and_voices(
    voices: list[str],
    model_dir: str | Path | None = None,
    voices_dir: str | Path | None = None,
    downloader: Callable[..., str] = hf_hub_download,
) -> tuple[Path, list[Path]]:
    """Install the configured model and voice files into local KTTS directories."""
    unique_voices = _validated_voices(voices)

    model_root = Path(model_dir or settings.model_dir).expanduser()
    voice_root = Path(voices_dir or settings.voices_dir).expanduser()
    model_relative = Path(model_config.model_file)
    model_destination = model_root / model_relative
    config_destination = model_destination.parent / MODEL_CONFIG_FILENAME
    voice_destinations = [voice_root / f"{voice}.pt" for voice in unique_voices]

    _download_asset(
        settings.model_repo_id,
        model_relative.name,
        model_destination,
        downloader,
    )
    _download_asset(
        settings.model_repo_id,
        MODEL_CONFIG_FILENAME,
        config_destination,
        downloader,
    )
    with config_destination.open(encoding="utf-8") as config_file:
        json.load(config_file)

    for voice, destination in zip(unique_voices, voice_destinations):
        _download_asset(
            settings.model_repo_id,
            f"voices/{voice}.pt",
            destination,
            downloader,
        )

    logger.info(f"Kokoro model installed at {model_destination}")
    logger.info(f"Installed {len(voice_destinations)} voice(s) in {voice_root}")
    return model_destination, voice_destinations
