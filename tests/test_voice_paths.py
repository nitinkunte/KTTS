import asyncio

import pytest

from ktts.core import paths


def test_voice_lookup_resolves_one_voice_by_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "af_heart.pt").touch()
    monkeypatch.setattr(paths.settings, "voices_dir", str(tmp_path))

    assert asyncio.run(paths.get_voice_path("af_heart")) == str(
        tmp_path / "af_heart.pt"
    )
    assert asyncio.run(paths.list_voices()) == ["af_heart"]


@pytest.mark.parametrize("voice_name", ("af_heart+af_bella", "../af_heart"))
def test_voice_lookup_rejects_combined_or_path_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    voice_name: str,
) -> None:
    (tmp_path / "af_heart.pt").touch()
    monkeypatch.setattr(paths.settings, "voices_dir", str(tmp_path))

    if "/" in voice_name:
        with pytest.raises(ValueError, match="must not include a path"):
            asyncio.run(paths.get_voice_path(voice_name))
    else:
        with pytest.raises(FileNotFoundError):
            asyncio.run(paths.get_voice_path(voice_name))
