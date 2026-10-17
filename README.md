# KTTS

KTTS is a CPU-only Kokoro text-to-speech library and command-line tool for
macOS. The library and flat `ktts` CLI support pause/resume and persistent
sessions. Implementation phases are documented in
[docs/New Project Plan.md](docs/New%20Project%20Plan.md).

KTTS uses one named voice per synthesis. Voice mixing, voice/rate tags,
aliases, SSML, HTTP APIs, web interfaces, and GPU inference are intentionally
out of scope. Voice names are looked up as individual `.pt` files under
`~/.cache/ktts/voices` by default.

## Development

This project uses `uv`:

```sh
uv sync
uv run pytest
```

The real-model integration test is skipped unless explicitly enabled and
`ktts setup` has installed the model and `af_heart` voice:

```sh
KTTS_RUN_INTEGRATION=1 uv run pytest -m integration
```

The Python API creates a synthesis plan once, saves rendered chunks as
individual WAV files, and joins them into `output.wav` when synthesis finishes:

```python
import asyncio
from ktts.engine import start


async def main():
    session = await start("Hello. [pause:1s] Welcome back.", voice="af_heart")
    async for audio_chunk in session.stream():
        pass
    print(session.output_path)


asyncio.run(main())
```

Call `session.pause()` to pause after the current planned chunk and
`session.resume()` to continue. `session.stop()` preserves completed chunks and
exports the partial WAV. To continue a saved session after a restart:

```python
from ktts import Session

session = await Session.restore(session_id)
session.resume()
async for audio_chunk in session.stream():
    pass
```

If a newer splitter produces a different plan, restore refuses by default;
choose `on_plan_mismatch="restart"` to start over or `"replan"` to keep
completed chunks and plan the remaining text again.

## CLI usage

Install Kokoro model weights and voice `.pt` files into the configured model
and voice directories before synthesis. By default, these are
`~/.cache/ktts/models` and `~/.cache/ktts/voices`.

Install the CLI with `uv`, then fetch the model configuration, model weights,
and default `af_heart` voice from the configured Hugging Face model repository:

```sh
uv tool install .
ktts setup
```

Install additional voices by repeating `--voice`; setup skips files that are
already present:

```sh
ktts setup --voice af_heart --voice bf_emma
```

To add voices later without touching the model, use `-d`/`--download`. It
fetches each `voices/NAME.pt` file (a small PyTorch voice-style tensor, one per
voice) from the `hexgrad/Kokoro-82M` repository and exits:

```sh
ktts -d bf_emma
ktts --download am_adam --download bf_isabella
```

Voices are never downloaded implicitly during synthesis. Selecting a voice that
is not installed fails immediately with a hint to run `ktts -d NAME`.

Run `ktts setup --help` for setup options. The model and voice directories can
be changed with the `MODEL_DIR` and `VOICES_DIR` environment variables. Downloads
are first retrieved through the Hugging Face Hub cache, then copied atomically
into KTTS's local directories. The Hub cache can be configured with its standard
Hugging Face environment variables.

```sh
ktts --help
ktts --listvoices
ktts "Hello. [pause:1s] Welcome back."
ktts --file notes.txt --voice af_heart --speed 0.9
ktts -f notes.txt -o speech.wav --force
ktts --sessions
ktts --resume
ktts --resume SESSION_ID
```

Use a positional string or `--file`, not both. Output defaults to the input
file's stem plus `.wav`, or a local-time timestamp for positional text. An
explicit non-WAV extension is replaced with `.wav` and reported. If `-o` is an
existing folder, or a path ending in `/` (created if missing), the default
name is used inside it: `ktts -f notes.txt -o ~/audio/` writes
`~/audio/notes.wav`. Existing
outputs are protected unless `--force` is given. `--speed` accepts 0.5 through
1.5. During interactive synthesis, Space toggles pause/resume and `q` saves
the session and exits; Ctrl-C also preserves progress.

`--strict` rejects phoneme symbols that the Kokoro vocabulary would drop.
Pronunciation overrides can be supplied with `--lexicon names.txt`; use one
`word = IPA` mapping per line and `#` to start a comment:

```text
# Pronunciations use symbols supported by the Kokoro vocabulary.
Quenthariel = kwɛnθɑɹiəl
```

Sessions are stored under `~/.local/share/ktts/sessions` by default. They
expire for resume after 14 days but are not automatically deleted. The
`SESSIONS_DIR`, `SESSION_TTL_DAYS`, `MAX_CONCURRENT_SESSIONS`, and
`PAUSE_ON_DISCONNECT` settings can be supplied as environment variables. Four
sessions may render concurrently per process by default. If an async consumer
closes or cancels a stream, the session auto-pauses by default. A pause request
allows the current planned chunk to finish first, so pause latency is bounded
by one chunk.

## eSpeak NG

Install eSpeak NG on macOS with Homebrew:

```sh
brew install espeak-ng
```

KTTS uses Misaki's bundled `espeakng-loader` data by default. If its data
cannot be found, set `ESPEAK_DATA_PATH` to Homebrew's data directory:

```sh
export ESPEAK_DATA_PATH="$(brew --prefix)/share/espeak-ng-data"
```

The startup check exercises an out-of-vocabulary English name and reports
installation or data-path problems before loading the Kokoro model.

## Attribution and licensing

Selected modules were copied or adapted from
[Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI), commit
`c8e000c6dbc5f136b6b82722d35b20b3b6f36564`. The copied project license is in
[LICENSE](LICENSE). Kokoro-FastAPI identifies its inference code as adapted
from StyleTTS2 and MIT-licensed; its non-inference wrapper code is Apache
License 2.0. The Kokoro model weights are separately licensed under Apache
License 2.0 by their authors. See the upstream
[Kokoro-FastAPI license notes](https://github.com/remsky/Kokoro-FastAPI#license)
and [Kokoro model page](https://huggingface.co/hexgrad/Kokoro-82M).
