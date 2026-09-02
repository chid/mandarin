#!/usr/bin/env python3
"""
Command-line interface for Ear Mandarin Pronunciation Evaluator.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

from ear_model import EarScorer, format_cli_output, sentence_to_pinyin_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Ear: Mandarin CTC Pronunciation Evaluator (On-device)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Command: evaluate
    eval_parser = subparsers.add_parser("eval", help="Evaluate an audio file")
    eval_parser.add_argument("audio_path", type=str, help="Path to audio file (wav, mp3, flac, etc.)")
    eval_parser.add_argument("--sentence", "-s", type=str, required=True, help="Target Chinese sentence, e.g. '你好朋友'")
    eval_parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    eval_parser.add_argument("--model", type=str, default="tiny_int8.onnx", help="Path to ONNX model")
    eval_parser.add_argument("--vocab", type=str, default="vocab.json", help="Path to vocab.json")

    # Command: pinyin
    pinyin_parser = subparsers.add_parser("pinyin", help="Convert Chinese text to target CTC pinyin tokens")
    pinyin_parser.add_argument("sentence", type=str, help="Target Chinese sentence")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Start local web server")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")

    args = parser.parse_args()

    if args.command == "pinyin":
        pairs = sentence_to_pinyin_tokens(args.sentence)
        print(f"Sentence: {args.sentence}")
        print("Tokens:")
        for char, tok in pairs:
            print(f"  {char} -> {tok}")
        return

    elif args.command == "serve":
        from server import run_server
        run_server(port=args.port)
        return

    elif args.command == "eval":
        audio_p = Path(args.audio_path)
        if not audio_p.exists():
            print(f"Error: Audio file '{args.audio_path}' not found.", file=sys.stderr)
            sys.exit(1)

        try:
            audio_data, sr = sf.read(str(audio_p))
        except Exception as e:
            print(f"Error reading audio file: {e}", file=sys.stderr)
            sys.exit(1)

        scorer = EarScorer(model_path=args.model, vocab_path=args.vocab)
        result = scorer.score_audio(audio_data, args.sentence, sample_rate=sr)

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_cli_output(result))
        return

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
