"""Command-line interface for local Kokoro synthesis."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import termios
import tty
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from huggingface_hub.errors import HfHubHTTPError

from .core import paths
from .core.config import settings
from .model_setup import download_voices, setup_model_and_voices
from .textproc.vocabulary import VOCAB, normalize_phonemes


def _load_session_api() -> None:
    if "Session" in globals():
        return
    from .session import (
        Session,
        SessionError,
        latest_unfinished_session_id,
        list_saved_sessions,
    )

    globals().update(
        Session=Session,
        SessionError=SessionError,
        latest_unfinished_session_id=latest_unfinished_session_id,
        list_saved_sessions=list_saved_sessions,
    )


def __getattr__(name: str):
    if name in {
        "Session",
        "SessionError",
        "latest_unfinished_session_id",
        "list_saved_sessions",
    }:
        _load_session_api()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("speed must be a number from 0.5 to 1.5") from error
    if not 0.5 <= parsed <= 1.5:
        raise argparse.ArgumentTypeError("speed must be between 0.5 and 1.5 inclusive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ktts",
        description="Generate a WAV file with Kokoro and pause/resume support.",
        epilog=(
            "During synthesis, press Space to pause/resume or q to save and quit. "
            "Run `ktts setup --help` to download model and voice assets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("-f", "--file", type=Path, help="Read text from a file")
    parser.add_argument(
        "-l", "--listvoices", action="store_true", help="List installed voices and exit"
    )
    parser.add_argument(
        "-v",
        "--voice",
        default=settings.default_voice,
        help="Voice name",
    )
    parser.add_argument(
        "-d",
        "--download",
        action="append",
        metavar="VOICE",
        help="Download a voice file (e.g. bf_emma) and exit; may be repeated",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=_positive_float,
        default=0.9,
        help="Speech speed from 0.5 to 1.5",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output WAV file, or a folder (existing, or ending in /) to hold the default-named file",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing output file"
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        metavar="ID",
        help="Resume a saved session by ID, or the most recent unfinished session",
    )
    parser.add_argument(
        "--sessions", action="store_true", help="List saved synthesis sessions and exit"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if synthesis would drop an unsupported phoneme symbol",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        help="Pronunciation overrides in `word = IPA` format",
    )
    return parser


def build_setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ktts setup",
        description="Download the Kokoro model and local voice files.",
        epilog="If --voice is omitted, the configured default voice is installed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--voice",
        action="append",
        dest="voices",
        help="Voice to install; may be repeated",
    )
    return parser


def _load_lexicon(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    lexicon: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected `word = IPA`")
        word, phonemes = (part.strip() for part in line.split("=", 1))
        if not word or not phonemes or any(character.isspace() for character in word):
            raise ValueError(
                f"{path}:{line_number}: expected one word and a non-empty IPA pronunciation"
            )
        if word.casefold() in {entry.casefold() for entry in lexicon}:
            raise ValueError(f"{path}:{line_number}: duplicate word {word!r}")
        phonemes = normalize_phonemes(phonemes, warn=False)
        unmapped = sorted(set(phonemes) - VOCAB.keys())
        if unmapped:
            raise ValueError(
                f"{path}:{line_number}: unsupported IPA symbols {unmapped!r}"
            )
        lexicon[word] = phonemes
    return lexicon


def _resolve_output_path(
    output: Optional[Path | str], input_file: Optional[Path]
) -> tuple[Path, Optional[str]]:
    default_name = (
        f"{input_file.stem}.wav"
        if input_file is not None
        else datetime.now().strftime("%Y%m%d%H%M%S.wav")
    )
    if output is not None:
        raw = os.fspath(output)
        directory = Path(raw).expanduser()
        # Path() drops a trailing slash, so check the raw string to detect a new folder.
        if raw.endswith(("/", os.sep)) or directory.is_dir():
            return (directory / default_name).resolve(), None
        output = Path(raw)
        if output.suffix.lower() == ".wav":
            return output.expanduser().resolve(), None
        return (
            output.with_suffix(".wav").expanduser().resolve(),
            f"Only WAV output is available; writing {output.with_suffix('.wav')}.",
        )
    return Path(default_name).resolve(), None


async def _run_session(session: Session, interactive: bool) -> None:
    key_task = None
    if interactive:
        key_task = asyncio.create_task(_watch_terminal_keys(session))
        print("Space: pause/resume | q: save and quit", file=sys.stderr, flush=True)
    try:
        async for _ in session.stream():
            chunk_index = max(0, session.cursor - 1)
            chunk = session.chunks[chunk_index]
            text = chunk.text if chunk.text else f"[pause: {chunk.pause_duration_s:g}s]"
            print(f"chunk {chunk_index + 1}/{len(session.chunks)}: {text}", flush=True)
    finally:
        if key_task is not None:
            key_task.cancel()
            try:
                await key_task
            except asyncio.CancelledError:
                pass
    if session.state == "stopped":
        print(f"Session {session.id} saved at chunk {session.cursor}.", file=sys.stderr)
    else:
        print(f"Audio written to {session.output_path}")


async def _watch_terminal_keys(session: Session) -> None:
    file_descriptor = sys.stdin.fileno()
    original_settings = termios.tcgetattr(file_descriptor)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()

    def read_key() -> None:
        character = os.read(file_descriptor, 1).decode(errors="ignore")
        if character:
            queue.put_nowait(character)

    try:
        tty.setcbreak(file_descriptor)
        loop.add_reader(file_descriptor, read_key)
        while True:
            key = await queue.get()
            if key == " ":
                if not session.is_paused:
                    session.pause()
                    print("Pausing after the current chunk.", file=sys.stderr, flush=True)
                else:
                    session.resume()
                    print("Resumed.", file=sys.stderr, flush=True)
            elif key.lower() == "q":
                session.stop()
                return
    finally:
        loop.remove_reader(file_descriptor)
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, original_settings)


def _validate_input(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.download:
        if (
            args.listvoices
            or args.sessions
            or args.text is not None
            or args.file is not None
            or args.resume is not None
        ):
            parser.error("--download cannot be combined with other actions or input")
        return
    if args.listvoices or args.sessions:
        if args.listvoices and args.sessions:
            parser.error("--listvoices and --sessions cannot be used together")
        if args.text is not None or args.file is not None or args.resume is not None:
            parser.error("listing options cannot be combined with synthesis input or --resume")
        return
    if args.resume is not None:
        if args.text is not None or args.file is not None:
            parser.error("--resume cannot be combined with text or --file")
        if (
            args.output is not None
            or args.lexicon is not None
            or args.strict
            or args.force
            or args.voice != settings.default_voice
            or args.speed != 0.9
        ):
            parser.error("--resume uses the saved output and synthesis options")
        return
    if args.text is not None and args.file is not None:
        parser.error("provide input text or --file, not both")
    if args.text is None and args.file is None:
        parser.error("provide input text or --file")


def _print_sessions() -> None:
    _load_session_api()
    records = list_saved_sessions()
    if not records:
        print("No saved sessions.")
        return
    for record in records:
        print(
            f"{record['id']}  {record['state']}  "
            f"chunk {record['cursor']}/{record['total']}  "
            f"{record['updated_at']}"
        )


async def _main_async(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.download:
        paths_installed = await asyncio.to_thread(download_voices, args.download)
        for path in paths_installed:
            print(f"Voice available at {path}")
        return
    if args.listvoices:
        voice_names = await paths.list_voices()
        if not voice_names:
            print(f"No voices found in {paths.voices_dir()}.")
            return
        print("\n".join(voice_names))
        return
    if args.sessions:
        _print_sessions()
        return
    if args.resume is not None:
        _load_session_api()
        session_id = args.resume or latest_unfinished_session_id()
        session = await Session.restore(session_id)
        session.resume()
        await _run_session(session, interactive=sys.stdin.isatty())
        return

    input_file = args.file
    if input_file is not None:
        text = input_file.read_text(encoding="utf-8")
    else:
        text = args.text

    installed_voices = await paths.list_voices()
    if args.voice not in installed_voices:
        parser.error(
            f"Voice {args.voice!r} is not installed. Run `ktts -l` to list installed "
            f"voices or `ktts -d {args.voice}` to download it."
        )

    output_path, notice = _resolve_output_path(args.output, input_file)
    if notice:
        print(notice, file=sys.stderr)
    if output_path.exists() and not args.force:
        parser.error(
            f"Output file already exists: {output_path}. Use --force to overwrite it."
        )
    lexicon = _load_lexicon(args.lexicon)
    _load_session_api()
    session = await Session.create(
        text,
        voice=args.voice,
        speed=args.speed,
        output_path=output_path,
        lexicon=lexicon,
        strict=args.strict,
    )
    print(f"Session {session.id}", file=sys.stderr, flush=True)
    await _run_session(session, interactive=sys.stdin.isatty())


def _setup_main(argv: Optional[Sequence[str]]) -> None:
    parser = build_setup_parser()
    args = parser.parse_args(argv)
    try:
        setup_model_and_voices(args.voices or [settings.default_voice])
    except (HfHubHTTPError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "setup":
        _setup_main(arguments[1:])
        return
    parser = build_parser()
    args = parser.parse_args(arguments)
    _validate_input(parser, args)
    _load_session_api()

    try:
        asyncio.run(_main_async(args, parser))
    except KeyboardInterrupt:
        print("\nInterrupted. The current session is saved and can be resumed with `ktts --resume`.", file=sys.stderr)
    except (HfHubHTTPError, OSError, SessionError, ValueError, RuntimeError) as error:
        parser.error(str(error))
