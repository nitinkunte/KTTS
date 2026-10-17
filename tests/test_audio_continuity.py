import asyncio
import io
import wave
from pathlib import Path

import av
import numpy as np
import pytest

from ktts.audio.streaming_audio_writer import StreamingAudioWriter
from ktts.base import AudioChunk
from ktts.engine import PlannedChunk
from ktts.session import Session


def _planned_text(*texts: str) -> list[PlannedChunk]:
    return [
        PlannedChunk(text=text, tokens=(1,), phonemes="həˈloʊ") for text in texts
    ]


def _renderer():
    async def render(chunk: PlannedChunk):
        # Different deterministic audio per text, including a boundary transition.
        seed = sum(chunk.text.encode("utf-8")) % 1000
        samples = np.sin(np.arange(6000, dtype=np.float32) / 23 + seed) * 0.2
        yield AudioChunk(samples)

    return render


def _wav_samples(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as audio:
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16)


def _encode_mp3(chunks: list[np.ndarray]) -> bytes:
    writer = StreamingAudioWriter("mp3", 24000)
    encoded = b"".join(writer.write_chunk(chunk) for chunk in chunks)
    encoded += writer.write_chunk(finalize=True)
    return encoded


def _decode_mp3(encoded: bytes) -> np.ndarray:
    with av.open(io.BytesIO(encoded), mode="r", format="mp3") as container:
        frames = [
            frame.to_ndarray().reshape(-1)
            for frame in container.decode(audio=0)
        ]
    return np.concatenate(frames) if frames else np.array([], dtype=np.int16)


def test_pause_resume_is_sample_identical_to_uninterrupted_wav_and_mp3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    planned = _planned_text("One.", "Two.", "Three.")

    async def build_plan(*args, **kwargs):
        return planned

    async def voice_path(_: str) -> str:
        return "/fake/af_heart.pt"

    monkeypatch.setattr("ktts.session.plan", build_plan)
    monkeypatch.setattr("ktts.session.paths.get_voice_path", voice_path)

    async def run_uninterrupted():
        session = await Session.create(
            "One. Two. Three.",
            session_root=tmp_path / "uninterrupted",
            renderer=_renderer(),
        )
        chunks = [chunk.audio.copy() async for chunk in session.stream()]
        return session, chunks

    async def run_paused():
        session = await Session.create(
            "One. Two. Three.",
            session_root=tmp_path / "paused",
            renderer=_renderer(),
        )
        stream = session.stream()
        first = await anext(stream)
        session.pause()
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        session.resume()
        remaining = [await pending]
        remaining.extend([chunk async for chunk in stream])
        return session, [first.audio.copy(), *(chunk.audio.copy() for chunk in remaining)]

    uninterrupted, baseline_chunks = asyncio.run(run_uninterrupted())
    resumed, resumed_chunks = asyncio.run(run_paused())
    baseline_pcm = np.concatenate(baseline_chunks)
    resumed_pcm = np.concatenate(resumed_chunks)

    assert np.array_equal(resumed_pcm, baseline_pcm)
    assert np.array_equal(_wav_samples(resumed.output_path), baseline_pcm)
    assert _wav_samples(resumed.output_path).tobytes() == _wav_samples(
        uninterrupted.output_path
    ).tobytes()

    baseline_mp3 = _encode_mp3(baseline_chunks)
    resumed_mp3 = _encode_mp3(resumed_chunks)
    assert resumed_mp3 == baseline_mp3
    decoded = _decode_mp3(resumed_mp3)
    assert decoded.size >= baseline_pcm.size - 2304

    # The chunk-edge jump is exactly the same for paused and uninterrupted streams.
    boundaries = np.cumsum([len(chunk) for chunk in baseline_chunks[:-1]])
    baseline_jumps = np.abs(baseline_pcm[boundaries] - baseline_pcm[boundaries - 1])
    resumed_jumps = np.abs(resumed_pcm[boundaries] - resumed_pcm[boundaries - 1])
    assert np.array_equal(resumed_jumps, baseline_jumps)
