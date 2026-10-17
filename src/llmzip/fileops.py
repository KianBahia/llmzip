"""File-level helpers shared by the CLI and the TUI: output naming and atomic writes."""

import os
import tempfile
from pathlib import Path

from llmzip.errors import LlmzipError, UsageError


def default_compress_output(src: Path) -> Path:
    return src.with_name(src.name + ".lz")


def default_decompress_output(src: Path) -> Path:
    if src.name.endswith(".lz") and len(src.name) > 3:
        return src.with_name(src.name[:-3])
    raise UsageError(f"{src} does not end in .lz; give an output name with -o")


def check_output(src: Path, out: Path, force: bool) -> None:
    if out.resolve() == src.resolve():
        raise UsageError(f"output {out} is the same file as the input")
    if out.exists() and not force:
        raise LlmzipError(f"{out} already exists (use --force to overwrite)")


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
