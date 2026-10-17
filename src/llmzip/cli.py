"""Command-line entry point: `uv run llmzip <command>`.

Exit codes: 0 ok, 1 error, 2 usage error. Set LLMZIP_DEBUG=1 for tracebacks.
"""

import argparse
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from llmzip import __version__, config
from llmzip.errors import LlmzipError


class UsageError(Exception):
    pass


# --- helpers ----------------------------------------------------------------------


def write_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file in the same directory + rename; never leaves a partial file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def check_output(src: Path, out: Path, force: bool) -> None:
    if out.resolve() == src.resolve():
        raise UsageError(f"output {out} is the same file as the input")
    if out.exists() and not force:
        raise LlmzipError(f"{out} already exists (use --force to overwrite)")


@contextmanager
def progress_factory(label: str, enabled: bool):
    """Yield a factory(total) -> advance(n) callback backed by a rich bar on stderr, or None."""
    if not enabled:
        yield None
        return
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        ProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text

    class SpeedColumn(ProgressColumn):
        def render(self, task):
            return Text(f"{task.speed:.0f} tok/s" if task.speed else "… tok/s")

    progress = Progress(
        TextColumn(label), BarColumn(), MofNCompleteColumn(), TextColumn("tok"),
        SpeedColumn(), TimeRemainingColumn(), console=Console(stderr=True), transient=True,
    )

    def factory(total: int):
        task = progress.add_task(label, total=total)
        return lambda n: progress.advance(task, n)

    with progress:
        yield factory


def show_progress(args) -> bool:
    return not args.quiet and sys.stderr.isatty()


# --- commands ---------------------------------------------------------------------


def cmd_compress(args: argparse.Namespace) -> int:
    src = Path(args.input)
    out = Path(args.output) if args.output else src.with_name(src.name + ".lz")
    check_output(src, out, args.force)
    data = src.read_bytes()

    from llmzip import api
    from llmzip.container import LLM, MODE_NAMES

    t0 = time.perf_counter()
    with progress_factory("compress", show_progress(args)) as pf:
        blob, info = api.compress_bytes(data, args.mode, args.block_size, pf, skip=not args.no_skip)
    if args.verify:
        with progress_factory("verify", show_progress(args)) as pf:
            restored, _ = api.decompress_bytes(blob, pf)
        if restored != data:
            raise LlmzipError("verification failed: decompressed output differs from input")
    write_atomic(out, blob)
    elapsed = time.perf_counter() - t0

    if not args.quiet:
        ratio = len(data) / len(blob)
        bpc = f", {len(blob) * 8 / info.chars:.3f} bits/char" if info.mode == LLM and info.chars else ""
        verified = ", verified" if args.verify else ""
        if info.llm_skipped:
            est, zst = info.llm_skipped
            verified += f"; LLM skipped: est. {est} B > zstd {zst} B"
        print(
            f"compress: {src} -> {out}  mode={MODE_NAMES[info.mode]}  "
            f"{len(data)} -> {len(blob)} B  (ratio {ratio:.2f}x{bpc}{verified})  {elapsed:.1f} s"
        )
    return 0


def cmd_decompress(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if args.output:
        out = Path(args.output)
    elif src.name.endswith(".lz") and len(src.name) > 3:
        out = src.with_name(src.name[:-3])
    else:
        raise UsageError(f"{src} does not end in .lz; give an output name with -o")
    check_output(src, out, args.force)
    blob = src.read_bytes()

    from llmzip import api
    from llmzip.container import MODE_NAMES

    t0 = time.perf_counter()
    with progress_factory("decompress", show_progress(args)) as pf:
        data, c = api.decompress_bytes(blob, pf)  # size + CRC32 verified in memory
    write_atomic(out, data)
    elapsed = time.perf_counter() - t0
    if not args.quiet:
        print(
            f"decompress: {src} -> {out}  mode={MODE_NAMES[c.mode]}  "
            f"{len(blob)} -> {len(data)} B  (CRC32 ok)  {elapsed:.1f} s"
        )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import llmzip.model  # noqa: F401  (first, so it can enable offline mode before transformers loads)
    import numpy
    import torch
    import transformers

    from llmzip.probs import PROBS_SCHEME

    torch.set_num_threads(config.NUM_THREADS)
    print(f"llmzip {__version__}")
    print(f"model:            {config.MODEL_NAME} @ {config.MODEL_REVISION}")
    print(f"attention:        {config.ATTN_IMPLEMENTATION}")
    print(f"BOS token id:     {config.BOS_TOKEN_ID}")
    print(f"max block tokens: {config.MAX_BLOCK_TOKENS}")
    print(f"NUM_THREADS:      {config.NUM_THREADS}")
    print(f"torch threads:    {torch.get_num_threads()}")
    print(f"FREQ_TOTAL_BITS:  {config.FREQ_TOTAL_BITS}")
    print(f"coder precision:  {config.CODER_PRECISION_BITS} bits")
    print(f"probs scheme:     {PROBS_SCHEME}")
    print(f"torch:            {torch.__version__}")
    print(f"transformers:     {transformers.__version__}")
    print(f"numpy:            {numpy.__version__}")
    if args.fingerprint:
        from llmzip.model import NextTokenModel

        print(f"fingerprint:      {NextTokenModel().model_fingerprint()}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from llmzip import bench

    paths = [Path(f) for f in args.files]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise UsageError(f"not a file: {', '.join(missing)}")
    results = bench.run(paths, args.block_sizes, not args.no_decompress, Path(args.out))
    failed = [r["file"] for r in results["files"] if r["llm"] and r["llm"]["roundtrip"] is False]
    if failed:
        raise LlmzipError(f"round trip FAILED for: {', '.join(failed)}")
    return 0


def cmd_not_implemented(args: argparse.Namespace) -> int:
    print(f"{args.command}: not implemented yet", file=sys.stderr)
    return 2


def block_size_arg(value: str) -> int:
    n = int(value)
    if not 1 <= n <= config.MAX_BLOCK_TOKENS:
        raise argparse.ArgumentTypeError(f"must be in 1..{config.MAX_BLOCK_TOKENS}")
    return n


def block_sizes_arg(value: str) -> list[int]:
    try:
        sizes = [block_size_arg(v) for v in value.split(",") if v.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    if not sizes:
        raise argparse.ArgumentTypeError("need at least one block size")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmzip", description="Lossless text compression with GPT-2.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compress", help="compress a file to FILE.lz")
    p.add_argument("input")
    p.add_argument("-o", "--output", help="output path (default: INPUT.lz)")
    p.add_argument("--mode", choices=["auto", "llm", "zstd", "store"], default="auto")
    p.add_argument("--block-size", type=block_size_arg, default=config.MAX_BLOCK_TOKENS,
                   help=f"max tokens per LLM block (1..{config.MAX_BLOCK_TOKENS})")
    p.add_argument("--force", action="store_true", help="overwrite the output if it exists")
    p.add_argument("--verify", action="store_true", help="decompress in-process and compare")
    p.add_argument("--no-skip", action="store_true",
                   help="auto mode: always run the full LLM, even if the first block predicts zstd wins")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress bar or summary")
    p.set_defaults(func=cmd_compress)

    p = sub.add_parser("decompress", help="decompress a .lz file")
    p.add_argument("input")
    p.add_argument("-o", "--output", help="output path (default: INPUT without .lz)")
    p.add_argument("--force", action="store_true", help="overwrite the output if it exists")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress bar or summary")
    p.set_defaults(func=cmd_decompress)

    p = sub.add_parser("bench", help="benchmark llmzip against gzip/xz/zstd")
    p.add_argument("files", nargs="+")
    p.add_argument("--block-sizes", type=block_sizes_arg, default=[config.MAX_BLOCK_TOKENS],
                   help="comma-separated LLM block sizes to compare (compress only)")
    p.add_argument("--no-decompress", action="store_true", help="skip the decompression round trip")
    p.add_argument("--out", default="bench/results", help="directory for the .json and .md results")
    p.set_defaults(func=cmd_bench)

    for name, help_text in [
        ("inspect", "show per-token surprise as a heatmap"),
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
    debug = os.environ.get("LLMZIP_DEBUG") == "1"
    try:
        return args.func(args)
    except UsageError as e:
        if debug:
            raise
        print(f"llmzip: usage error: {e}", file=sys.stderr)
        return 2
    except (LlmzipError, OSError, ValueError) as e:
        if debug:
            raise
        print(f"llmzip: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("llmzip: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
