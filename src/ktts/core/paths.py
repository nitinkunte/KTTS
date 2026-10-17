"""Local model and voice file lookup."""

from pathlib import Path

import aiofiles.os

from .config import settings


def models_dir() -> str:
    return settings.model_dir


def voices_dir() -> str:
    return settings.voices_dir


async def _find_file(filename: str, directory: str) -> str:
    root = Path(directory).expanduser().resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"File name must not include a path: {filename!r}"
        ) from error
    if await aiofiles.os.path.isfile(candidate):
        return str(candidate)
    raise FileNotFoundError(f"File not found: {candidate}")


async def get_model_path(model_name: str) -> str:
    try:
        return await _find_file(model_name, models_dir())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}. The Kokoro model is not installed; run `ktts setup` to download it."
        ) from error


async def get_voice_path(voice_name: str) -> str:
    filename = voice_name if voice_name.endswith(".pt") else f"{voice_name}.pt"
    if Path(filename).name != filename:
        raise ValueError(f"Voice name must not include a path: {voice_name!r}")
    return await _find_file(filename, voices_dir())


async def list_voices() -> list[str]:
    voice_dir = Path(voices_dir()).expanduser()
    if not await aiofiles.os.path.isdir(voice_dir):
        return []
    return sorted(
        entry.name.removesuffix(".pt")
        for entry in await aiofiles.os.scandir(voice_dir)
        if entry.is_file() and entry.name.endswith(".pt")
    )
