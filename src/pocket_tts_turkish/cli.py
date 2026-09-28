"""Command line tool: generate, normalize, voices, cut-reference."""

import argparse
import sys
import warnings
from pathlib import Path

from . import __version__
from .audio import cut_reference, load_audio, save_wav
from .model import DEFAULT_REPO, TurkishTTS, _model_folder, _natural_key
from .normalize import normalize

_RATE = 24000


def _read_text(args: argparse.Namespace) -> str:
    """Text from --text, or from --text-file ("-" reads standard input)."""
    if args.text is not None:
        return args.text
    if args.text_file == "-":
        return sys.stdin.read()
    return Path(args.text_file).read_text(encoding="utf-8")


def _generate(args: argparse.Namespace) -> int:
    """Speak the text into a WAV file."""
    tts = TurkishTTS.from_pretrained(args.model, device=args.device)
    voice = tts.voice_from_file(args.voice_file) if args.voice_file else args.voice
    audio = tts.generate(_read_text(args), voice=voice, emotion=args.emotion, seed=args.seed)
    path = save_wav(args.output, audio, tts.sample_rate)
    print(f"{path} ({len(audio) / tts.sample_rate:.1f} s)")
    return 0


def _normalize(args: argparse.Namespace) -> int:
    """Print the text with numbers, units and symbols written out."""
    print(normalize(_read_text(args)))
    return 0


def _voices(args: argparse.Namespace) -> int:
    """List the voices bundled with the model."""
    folder = _model_folder(args.model, None, None)
    print("\n".join(sorted((p.stem for p in (folder / "voices").glob("*.wav")), key=_natural_key)))
    return 0


def _cut(args: argparse.Namespace) -> int:
    """Cut a recording into a voice reference; exit code 2 when no pause was found."""
    wav, found = cut_reference(load_audio(args.input, _RATE), _RATE, args.min, args.max)
    path = save_wav(args.output, wav, _RATE)
    note = "" if found else ", no pause between words found: listen before using it"
    print(f"{path} ({len(wav) / _RATE:.2f} s{note})")
    return 0 if found else 2


def _add_text_args(parser: argparse.ArgumentParser) -> None:
    """--text or --text-file, one of them required."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="text to read")
    group.add_argument("--text-file", help="UTF-8 file with the text, or - for standard input")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for all subcommands."""
    parser = argparse.ArgumentParser(prog="pocket-tts-turkish", description="Turkish text-to-speech on Pocket TTS.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="speak text into a WAV file")
    _add_text_args(gen)
    voice = gen.add_mutually_exclusive_group()
    voice.add_argument("--voice", help="bundled voice (default: the first one, see 'voices')")
    voice.add_argument("--voice-file", help="your own recording to clone (cut automatically)")
    gen.add_argument("--emotion", help="mutlu, üzgün, kızgın, şaşkın, sakin or their English names")
    gen.add_argument("--seed", type=int, help="make the output repeatable")
    gen.add_argument("-o", "--output", default="output.wav", help="output WAV file (default: output.wav)")
    gen.add_argument("--model", default=DEFAULT_REPO, help=f"Hugging Face repo id or local folder (default: {DEFAULT_REPO})")
    gen.add_argument("--device", default="auto", help="auto, cpu or cuda (default: auto)")
    gen.set_defaults(func=_generate)

    norm = sub.add_parser("normalize", help="print the text as the model will read it")
    _add_text_args(norm)
    norm.set_defaults(func=_normalize)

    voices = sub.add_parser("voices", help="list the bundled voices")
    voices.add_argument("--model", default=DEFAULT_REPO, help="Hugging Face repo id or local folder")
    voices.set_defaults(func=_voices)

    cut = sub.add_parser("cut-reference", help="cut a recording into a voice reference")
    cut.add_argument("input", help="recording of the voice (WAV, FLAC, OGG or MP3)")
    cut.add_argument("output", help="output WAV file")
    cut.add_argument("--min", type=float, default=3.0, help="earliest cut point in seconds (default: 3)")
    cut.add_argument("--max", type=float, default=5.0, help="latest cut point in seconds (default: 5)")
    cut.set_defaults(func=_cut)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line tool and return its exit code."""
    args = build_parser().parse_args(argv)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            return args.func(args)
    except (FileNotFoundError, TypeError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
