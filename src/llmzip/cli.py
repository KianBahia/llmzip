"""Command-line entry point: `uv run llmzip <command>`."""

import argparse
import sys

from llmzip import __version__, config


def cmd_info(args: argparse.Namespace) -> int:
    import torch
    import transformers

    torch.set_num_threads(config.NUM_THREADS)
    print(f"llmzip {__version__}")
    print(f"model:            {config.MODEL_NAME} @ {config.MODEL_REVISION}")
    print(f"attention:        {config.ATTN_IMPLEMENTATION}")
    print(f"BOS token id:     {config.BOS_TOKEN_ID}")
    print(f"NUM_THREADS:      {config.NUM_THREADS}")
    print(f"torch threads:    {torch.get_num_threads()}")
    print(f"FREQ_TOTAL_BITS:  {config.FREQ_TOTAL_BITS}")
    print(f"torch:            {torch.__version__}")
    print(f"transformers:     {transformers.__version__}")
    if args.fingerprint:
        from llmzip.model import NextTokenModel

        print(f"fingerprint:      {NextTokenModel().model_fingerprint()}")
    return 0


def cmd_not_implemented(args: argparse.Namespace) -> int:
    print(f"{args.command}: not implemented yet", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmzip", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("compress", "compress a file"),
        ("decompress", "decompress a .lz file"),
        ("inspect", "show per-token surprise as a heatmap"),
        ("bench", "benchmark against gzip/xz/zstd"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("args", nargs="*")
        p.set_defaults(func=cmd_not_implemented)

    p = sub.add_parser("info", help="print configuration and versions")
    p.add_argument("--fingerprint", action="store_true", help="load the model and print its fingerprint")
    p.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
