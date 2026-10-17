"""End-to-end CLI demo: compress, decompress in a separate process, cmp.

Run: uv run python scripts/demo_cli.py FILE [FILE ...]
With no arguments: every file in bench/samples/, 64 KiB of random bytes, and a PNG
from /System/Library.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from llmzip import container as ct

ROOT = Path(__file__).resolve().parents[1]
PREFERRED_PNG = Path("/System/Library/Desktop Pictures/.wallpapers/Sequoia Sunrise/Sequoia Sunrise Thumbnail.png")


def find_png() -> Path | None:
    if PREFERRED_PNG.is_file():
        return PREFERRED_PNG
    out = subprocess.run(
        ["find", "/System/Library", "-name", "*.png", "-size", "+20k", "-size", "-500k"],
        capture_output=True, text=True,
    )
    hits = sorted(out.stdout.splitlines())
    return Path(hits[0]) if hits else None


def llmzip(*args) -> float:
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "-m", "llmzip.cli", *map(str, args)], check=True,
                   capture_output=True, text=True)
    return time.perf_counter() - t0


def gzip_size(path: Path) -> int:
    return len(subprocess.run(["gzip", "-9", "-c", str(path)], capture_output=True, check=True).stdout)


def main(paths: list[Path]) -> int:
    rows, ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for path in paths:
            lz, out = tmp / (path.name + ".lz"), tmp / (path.name + ".out")
            t_c = llmzip("compress", path, "-o", lz, "--force", "--quiet")
            t_d = llmzip("decompress", lz, "-o", out, "--force", "--quiet")
            same = subprocess.run(["cmp", "-s", str(path), str(out)]).returncode == 0
            ok &= same
            mode = ct.MODE_NAMES[ct.unpack(lz.read_bytes()).mode]
            orig, comp = path.stat().st_size, lz.stat().st_size
            rows.append((path.name, mode, orig, comp, orig / comp, gzip_size(path), t_c, t_d, same))

    print(f"{'file':<34} {'mode':<6} {'orig':>7} {'.lz':>7} {'ratio':>6} {'gzip-9':>7} "
          f"{'comp s':>7} {'decomp s':>8}  cmp")
    for name, mode, orig, comp, ratio, gz, t_c, t_d, same in rows:
        print(f"{name[:34]:<34} {mode:<6} {orig:>7} {comp:>7} {ratio:>5.2f}x {gz:>7} "
              f"{t_c:>7.1f} {t_d:>8.1f}  {'OK' if same else 'DIFFER'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = [Path(p) for p in sys.argv[1:]]
    else:
        files = sorted((ROOT / "bench" / "samples").iterdir())
        rnd = Path(tempfile.gettempdir()) / "llmzip_random_64k.bin"
        rnd.write_bytes(os.urandom(64 * 1024))
        files.append(rnd)
        png = find_png()
        if png:
            print(f"PNG used: {png}")
            files.append(png)
        else:
            print("no PNG found under /System/Library")
    sys.exit(main(files))
