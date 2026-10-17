"""Shared test helpers."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "bench" / "samples"

sys.path.insert(0, str(ROOT / "scripts"))
from smoke_model import PARAGRAPH  # noqa: E402,F401


def text_with_exactly(n_tokens: int, model) -> str:
    """Decode a prefix of a long text's ids so it re-tokenizes to exactly n_tokens."""
    ids = model.encode((PARAGRAPH + "\n\n") * (n_tokens // 200 + 2))
    for k in range(n_tokens, n_tokens - 20, -1):
        for extra in range(0, 20):
            text = model.decode(ids[: k + extra])
            if len(model.encode(text)) == n_tokens:
                return text
    raise AssertionError(f"could not build a {n_tokens}-token text")


def run_cli(*args, check: bool = True) -> subprocess.CompletedProcess:
    """Run `llmzip ...` in a fresh Python process."""
    return subprocess.run(
        [sys.executable, "-m", "llmzip.cli", *map(str, args)],
        capture_output=True, text=True, check=check,
    )
