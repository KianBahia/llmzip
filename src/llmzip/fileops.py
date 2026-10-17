"""File-level helpers shared by the CLI and the TUI: output naming and atomic writes."""

import hashlib
import os
import tempfile
from dataclasses import dataclass
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


# --- restored copies, unique names, identity check -------------------------------


def restored_output(lz: Path) -> Path:
    """Name for a restored copy: insert ".restored" before the original's LAST extension.

    report.md.lz -> report.restored.md, notes.lz -> notes.restored,
    a.tar.gz.lz -> a.tar.restored.gz, .bashrc.lz -> .bashrc.restored
    """
    inner = Path(default_decompress_output(lz).name)  # raises UsageError unless *.lz
    if inner.suffix:
        return lz.with_name(f"{inner.stem}.restored{inner.suffix}")
    return lz.with_name(f"{inner.name}.restored")


def _numbered(name: str, n: int, where: str) -> str:
    if where == "base":  # before ALL extensions: report.md.lz -> report-2.md.lz
        dots = len(name) - len(name.lstrip("."))
        base, sep, rest = name[dots:].partition(".")
        return f"{name[:dots]}{base}-{n}{sep}{rest}"
    p = Path(name)  # before the LAST extension: report.restored.md -> report.restored-2.md
    return f"{p.stem}-{n}{p.suffix}"


def unique_path(path: Path, where: str = "last") -> Path:
    """`path` if free, else the first free of name-2, name-3, … (numbered at `where`).

    where="last": before the last extension (used for restored copies).
    where="base": after the base name, before all extensions (used for .lz outputs, so
    report-2.md.lz still restores to report-2.restored.md).
    """
    if where not in ("last", "base"):
        raise ValueError(f"where must be 'last' or 'base', not {where!r}")
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(_numbered(path.name, n, where))
        if not candidate.exists():
            return candidate
        n += 1


@dataclass
class Identity:
    original: Path
    restored: Path
    original_size: int
    restored_size: int
    original_sha256: str
    restored_sha256: str

    @property
    def identical(self) -> bool:
        return self.original_size == self.restored_size and self.original_sha256 == self.restored_sha256

    def line(self) -> str:
        verdict, op = ("IDENTICAL ✓", "=") if self.identical else ("DIFFERENT ✗", "≠")
        return (f"Original vs restored: {verdict}  "
                f"{self.original.name} ({self.original_size:,} B, {self.original_sha256[:12]}) {op} "
                f"{self.restored.name} ({self.restored_size:,} B, {self.restored_sha256[:12]})")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity_check(lz: Path, restored: Path) -> Identity | None:
    """Compare a restored file with the original (the .lz name minus ".lz", next to the
    restored file). None if there is no such original, or it IS the restored file."""
    if not lz.name.endswith(".lz"):
        return None
    original = restored.parent / lz.name[:-3]
    if not original.is_file() or original.resolve() == restored.resolve():
        return None
    return Identity(original, restored, original.stat().st_size, restored.stat().st_size,
                    _sha256(original), _sha256(restored))


NOT_PRESENT = "Original not present — CRC32 verified."
