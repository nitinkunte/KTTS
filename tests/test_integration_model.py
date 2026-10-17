import asyncio
import os
from pathlib import Path

import pytest

from ktts.core.config import settings
from ktts.core.model_config import model_config
from ktts.engine import start

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("KTTS_RUN_INTEGRATION") != "1",
    reason="set KTTS_RUN_INTEGRATION=1 to run real Kokoro model inference",
)
def test_real_kokoro_model_generates_nonempty_wav(tmp_path: Path) -> None:
    model_path = Path(settings.model_dir).expanduser() / model_config.model_file
    voice_path = Path(settings.voices_dir).expanduser() / "af_heart.pt"
    if not model_path.is_file() or not voice_path.is_file():
        pytest.skip("run `ktts setup` to download the integration-test assets")

    async def generate():
        session = await start(
            "This is a real model integration test.",
            voice="af_heart",
            output_path=tmp_path / "real-model.wav",
        )
        async for _ in session.stream():
            pass
        return session

    session = asyncio.run(generate())
    assert session.output_path.is_file()
    assert session.output_path.stat().st_size > 44
