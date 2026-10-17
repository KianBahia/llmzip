"""`llmzip bench`: llmzip (forced LLM and auto) vs gzip -9 / xz -9 / zstd -19.

External tools read the file from stdin so no filename is stored in their output.
llmzip times exclude the one-off model load, which is reported separately.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from llmzip import api
from llmzip import container as ct
from llmzip.config import LLM_TOKEN_BUDGET, MAX_BLOCK_TOKENS
from llmzip.llmcodec import NotRoundTrippable, get_model, tokenize_checked

EXTERNAL = {
    "gzip -9": ["gzip", "-9", "-c"],
    "xz -9": ["xz", "-9", "-c"],
    "zstd -19": ["zstd", "-19", "-c", "-q"],
}


def _external_size(cmd: list[str], path: Path) -> int:
    with open(path, "rb") as f:
        return len(subprocess.run(cmd, stdin=f, capture_output=True, check=True).stdout)


def _bpb(size: int | None, n: int) -> float | None:
    return None if size is None or n == 0 else size * 8 / n


def _auto_choice(raw: bytes, llm_blob: bytes, model, block_size: int, ids: list[int]) -> dict:
    """What `compress --mode auto` picks, derived without re-running the LLM.

    The forced-LLM blocks are bit-identical to what auto mode would code, so the
    skip heuristic can be replayed from its first block.
    """
    stored, _ = api.compress_bytes(raw, mode="store")
    zstd, _ = api.compress_bytes(raw, mode="zstd")
    candidates = {ct.STORED: len(stored), ct.ZSTD: len(zstd), ct.LLM: len(llm_blob)}
    skipped = None
    if LLM_TOKEN_BUDGET and len(ids) > LLM_TOKEN_BUDGET:
        skipped = f"over budget ({len(ids)} > {LLM_TOKEN_BUDGET} tokens)"
        del candidates[ct.LLM]
    elif len(ids) > 2 * block_size:
        first = ct.unpack(llm_blob).body.blocks[0]
        estimate, skip = api.skip_estimate(first, ids[:block_size], model, len(raw), len(zstd) - ct.HEADER_SIZE)
        if skip:
            skipped = estimate
            del candidates[ct.LLM]
    mode = min(candidates, key=lambda m: (candidates[m], m))
    return {"mode": ct.MODE_NAMES[mode], "size": candidates[mode], "llm_skipped_estimate": skipped}


def bench_file(path: Path, model, block_sizes: list[int], decompress: bool, progress=None) -> dict:
    """progress: optional ProgressFactory for the forced-LLM compression (lets a UI cancel it)."""
    raw = path.read_bytes()
    row: dict = {"file": path.name, "bytes": len(raw)}
    for name, cmd in EXTERNAL.items():
        size = _external_size(cmd, path)
        row[name] = {"size": size, "bpb": _bpb(size, len(raw))}

    ids = None
    try:
        text = raw.decode("utf-8")
        ids = tokenize_checked(text, model) if raw else None
    except (UnicodeDecodeError, NotRoundTrippable):
        pass
    if not ids:
        row["llm"] = None
        blob, info = api.compress_bytes(raw, mode="auto", model=model)
        row["auto"] = {"mode": ct.MODE_NAMES[info.mode], "size": len(blob), "llm_skipped_estimate": None}
        return row

    t0 = time.perf_counter()
    blob, _ = api.compress_bytes(raw, mode="llm", model=model, progress=progress)
    t_c = time.perf_counter() - t0
    body = ct.unpack(blob).body
    payload = sum(len(b) for b in body.blocks)
    llm = {
        "tokens": len(ids),
        "blocks": len(body.blocks),
        "payload": payload,
        "size": len(blob),
        "bpb": _bpb(len(blob), len(raw)),
        "payload_bpb": _bpb(payload, len(raw)),
        "compress_s": t_c,
        "compress_tok_s": len(ids) / t_c,
        "decompress_s": None,
        "decompress_tok_s": None,
        "roundtrip": None,
    }
    if decompress:
        t0 = time.perf_counter()
        restored, _ = api.decompress_bytes(blob, model=model)
        t_d = time.perf_counter() - t0
        llm.update(decompress_s=t_d, decompress_tok_s=len(ids) / t_d, roundtrip=restored == raw)
    row["llm"] = llm
    row["auto"] = _auto_choice(raw, blob, model, MAX_BLOCK_TOKENS, ids)

    row["block_sizes"] = []
    for bs in block_sizes:
        if bs == MAX_BLOCK_TOKENS:
            size, pl, t = len(blob), payload, t_c
        else:
            t0 = time.perf_counter()
            b, _ = api.compress_bytes(raw, mode="llm", block_size=bs, model=model)
            t = time.perf_counter() - t0
            size, pl = len(b), sum(len(x) for x in ct.unpack(b).body.blocks)
        row["block_sizes"].append({
            "block_size": bs, "blocks": -(-len(ids) // bs), "size": size, "payload": pl,
            "bpb": _bpb(size, len(raw)), "compress_s": t,
        })
    return row


def _fmt(x, spec=".3f"):
    return "n/a" if x is None else format(x, spec)


def markdown(results: dict) -> str:
    lines = [
        f"# llmzip bench — {results['timestamp']}",
        "",
        f"Model load (incl. fingerprint): {results['model_load_s']:.2f} s. "
        "Sizes in bytes; bpb = bits per original byte. llmzip times exclude model load.",
        "",
        "| file | bytes | llmzip LLM | bpb | auto pick | gzip -9 | bpb | xz -9 | bpb | zstd -19 | bpb "
        "| comp s | decomp s | tok/s (c/d) | round trip |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results["files"]:
        llm = r["llm"]
        auto = f"{r['auto']['mode']} ({r['auto']['size']})"
        ext = " | ".join(f"{r[k]['size']} | {_fmt(r[k]['bpb'])}" for k in EXTERNAL)
        if llm:
            rt = {True: "OK", False: "FAIL", None: "skipped"}[llm["roundtrip"]]
            tok_s = f"{llm['compress_tok_s']:.0f} / {_fmt(llm['decompress_tok_s'], '.0f')}"
            lines.append(
                f"| {r['file']} | {r['bytes']} | {llm['size']} | {_fmt(llm['bpb'])} | {auto} | {ext} | "
                f"{llm['compress_s']:.1f} | {_fmt(llm['decompress_s'], '.1f')} | {tok_s} | {rt} |"
            )
        else:
            lines.append(f"| {r['file']} | {r['bytes']} | n/a | n/a | {auto} | {ext} | n/a | n/a | n/a | n/a |")
    sized = [r for r in results["files"] if r.get("block_sizes") and len(r["block_sizes"]) > 1]
    if sized:
        lines += ["", "## Block-size experiment (LLM mode, compress only)", "",
                  "| file | block size | blocks | .lz bytes | payload | bpb | comp s |", "|---|---|---|---|---|---|---|"]
        for r in sized:
            for b in r["block_sizes"]:
                lines.append(f"| {r['file']} | {b['block_size']} | {b['blocks']} | {b['size']} | "
                             f"{b['payload']} | {b['bpb']:.3f} | {b['compress_s']:.1f} |")
    return "\n".join(lines) + "\n"


def print_table(results: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"llmzip bench (model load {results['model_load_s']:.2f} s, excluded from times)")
    for col in ["file", "bytes", "LLM .lz", "bpb", "auto", "gzip-9", "xz-9", "zstd-19",
                "comp s", "decomp s", "round trip"]:
        table.add_column(col, justify="left" if col in ("file", "auto") else "right")
    for r in results["files"]:
        llm = r["llm"]
        ext = [f"{r[k]['size']} ({_fmt(r[k]['bpb'], '.2f')})" for k in EXTERNAL]
        auto = r["auto"]["mode"] + (" (LLM skipped)" if r["auto"]["llm_skipped_estimate"] else "")
        if llm:
            rt = {True: "OK", False: "FAIL", None: "-"}[llm["roundtrip"]]
            table.add_row(r["file"], str(r["bytes"]), str(llm["size"]), _fmt(llm["bpb"], ".2f"), auto, *ext,
                          f"{llm['compress_s']:.1f}", _fmt(llm["decompress_s"], ".1f"), rt)
        else:
            table.add_row(r["file"], str(r["bytes"]), "n/a", "n/a", auto, *ext, "n/a", "n/a", "n/a")
    console = Console()
    if not console.is_terminal:
        console = Console(width=160)  # don't truncate columns when piped
    console.print(table)


def run(paths: list[Path], block_sizes: list[int], decompress: bool, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    model = get_model()
    model.fingerprint_bytes()
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "model_load_s": time.perf_counter() - t0,
        "fingerprint": model.model_fingerprint(),
        "block_sizes": block_sizes,
        "files": [bench_file(p, model, block_sizes, decompress) for p in paths],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / results["timestamp"]
    stem.with_suffix(".json").write_text(json.dumps(results, indent=2))
    stem.with_suffix(".md").write_text(markdown(results))
    print_table(results)
    print(f"wrote {stem}.json and {stem}.md")
    return results
