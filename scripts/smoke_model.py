"""Phase 1 smoke test: determinism, speed, and ideal compressed size.

Run: uv run python scripts/smoke_model.py
"""

import math
import subprocess
import sys
import time

import torch

from llmzip import config
from llmzip.model import NextTokenModel, logits_digest

PARAGRAPH = (
    "The old lighthouse stood at the edge of the cliff for more than a century, "
    "guiding fishing boats and cargo ships safely past the rocks that lay hidden "
    "beneath the waves. Every evening, the keeper climbed the narrow spiral staircase, "
    "trimmed the wick, and polished the great glass lens until it gleamed. In the "
    "early years the light was fueled by whale oil, later by kerosene, and finally by "
    "electricity, when a cable was laid across the bay from the mainland. The keepers "
    "kept careful logs of the weather, the passing vessels, and the occasional "
    "shipwreck, and those logs are now stored in the town library, where historians "
    "read them to understand how life on the coast has changed. Today the lighthouse "
    "is automated, and no one lives in the small white cottage beside it. Tourists "
    "walk up the path in the summer to take photographs and look out over the water. "
    "On clear nights you can still see the beam sweeping across the sea, steady and "
    "patient, just as it has always been. For the people of the town, it remains a "
    "symbol of safety, of home, and of the quiet work that keeps others from harm, "
    "even when nobody is watching and nobody remembers to say thank you."
)


def performance_cores() -> int:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, check=True
        )
        return int(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return torch.get_num_threads()


def as_bytes(logits: list[torch.Tensor]) -> list[bytes]:
    return [t.numpy().tobytes() for t in logits]


def timed_run(model: NextTokenModel, ids: list[int]) -> tuple[list[bytes], float]:
    t0 = time.perf_counter()
    logits = model.logits_sequence(ids)
    return as_bytes(logits), time.perf_counter() - t0


def subprocess_digest(num_threads: int, text: str) -> str:
    out = subprocess.run(
        [sys.executable, "-m", "llmzip.model", str(num_threads)],
        input=text, capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def external_size(cmd: list[str], data: bytes) -> int:
    return len(subprocess.run(cmd, input=data, capture_output=True, check=True).stdout)


def main() -> int:
    model = NextTokenModel()
    ids = model.encode(PARAGRAPH)
    n_words = len(PARAGRAPH.split())
    n_chars = len(PARAGRAPH)
    print(f"paragraph: {n_words} words, {n_chars} chars, {len(ids)} tokens")
    print(f"fingerprint (threads={model.num_threads}): {model.model_fingerprint()}\n")

    thread_counts = sorted({1, 4, performance_cores()})
    results = {}
    reference = None
    for n in thread_counts:
        model.set_num_threads(n)
        model.logits_sequence(ids)  # warm-up
        run_a, dt_a = timed_run(model, ids)
        run_b, dt_b = timed_run(model, ids)
        same = run_a == run_b
        if reference is None:
            reference = run_a
        digest = logits_digest(model, PARAGRAPH)
        sub = [subprocess_digest(n, PARAGRAPH) for _ in range(2)]
        results[n] = {
            "in_process": same,
            "cross_threads": run_a == reference,
            "tok_s": len(ids) / min(dt_a, dt_b),
            "digest": digest,
            "subprocess": [d == digest for d in sub],
        }
        r = results[n]
        print(
            f"threads={n:>2}: in-process identical={same}  "
            f"identical to threads={thread_counts[0]}: {r['cross_threads']}  "
            f"{r['tok_s']:.1f} tok/s"
        )
        print(f"            parent digest {digest[:16]}...  fresh-process matches: {r['subprocess']}")

    ok = [n for n in thread_counts if results[n]["in_process"] and all(results[n]["subprocess"])]
    best = max(ok, key=lambda n: results[n]["tok_s"]) if ok else None
    print(f"\nrecommended NUM_THREADS: {best}  (fastest fully deterministic; config has {config.NUM_THREADS})")

    # Batched vs incremental (information only; batched is never used for coding).
    model.set_num_threads(config.NUM_THREADS)
    incremental = torch.stack(model.logits_sequence(ids))
    with torch.no_grad():
        batched = model.model(
            input_ids=torch.tensor([[config.BOS_TOKEN_ID] + ids[:-1]])
        ).logits[0]
    max_diff = (batched - incremental).abs().max().item()
    print(
        f"\nbatched vs incremental: bit-identical={torch.equal(batched, incremental)}  "
        f"max |diff|={max_diff:.3e}"
    )

    # Ideal compressed size: sum of -log2 p(actual token).
    logprobs = torch.log_softmax(incremental.double(), dim=-1)
    nats = -logprobs[torch.arange(len(ids)), torch.tensor(ids)].sum().item()
    bits = nats / math.log(2)
    raw = PARAGRAPH.encode("utf-8")
    print(f"\nideal size: {bits:.1f} bits = {bits / 8:.1f} bytes, {bits / n_chars:.3f} bits/char")
    print(f"raw:        {len(raw)} bytes, 8.000 bits/char")
    for name, cmd in [
        ("gzip -9", ["gzip", "-9", "-c"]),
        ("xz -9", ["xz", "-9", "-c"]),
        ("zstd -19", ["zstd", "-19", "-c", "-q"]),
    ]:
        size = external_size(cmd, raw)
        print(f"{name:<10}  {size} bytes, {size * 8 / n_chars:.3f} bits/char  (includes format header)")
    return 0 if best is not None else 1


if __name__ == "__main__":
    sys.exit(main())
