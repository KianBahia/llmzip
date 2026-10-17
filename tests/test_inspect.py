import json
import math

import pytest

from llmzip import api
from llmzip import container as ct
from llmzip.inspect_cmd import analyze, autopilot_runs, render, top_surprising
from llmzip.llmcodec import compress_text
from llmzip.model import NextTokenModel

from helpers import PARAGRAPH, SAMPLES, run_cli

PROSE = (SAMPLES / "prose.txt").read_bytes().decode("utf-8")


@pytest.fixture(scope="module")
def model():
    return NextTokenModel()


@pytest.fixture(scope="module")
def prose(model):
    return analyze(PROSE, model)


def test_predicted_payload_matches_per_block_bits_and_actual(prose, model):
    per_block: dict[int, float] = {}
    for t in prose.tokens:
        per_block[t.block] = per_block.get(t.block, 0.0) + t.quant_bits
    assert prose.predicted_payload == sum(math.ceil(b / 8) for b in per_block.values())
    assert abs(prose.predicted_payload - prose.actual_payload) <= 4
    # inspect's in-pass coding is the real compression output, byte for byte
    blocks, _ = compress_text(PROSE, model)
    assert prose.block_payloads == blocks


def test_multi_block_matches_compression(model):
    a = analyze(PROSE, model, block_size=50)
    blocks, n = compress_text(PROSE, model, block_size=50)
    assert len(a.block_payloads) == len(blocks) == -(-n // 50)
    assert a.block_payloads == blocks
    assert {t.block for t in a.tokens} == set(range(len(blocks)))
    assert abs(a.predicted_payload - a.actual_payload) <= 2 * len(blocks)


def test_segments_reproduce_text(model):
    text = "emoji 🍕🔥 and 中文, café\r\n\tend"
    a = analyze(text, model, block_size=3)
    assert "".join(t.text for t in a.tokens) == text


def test_ideal_bits_match_phase1_smoke_paragraph(model):
    a = analyze(PARAGRAPH, model)
    assert a.ideal_bits == pytest.approx(1151.8, abs=0.1)


def test_top_n_sorted_and_top3_valid(prose, model):
    idx = top_surprising(prose, 10)
    bits = [prose.tokens[i].quant_bits for i in idx]
    assert len(idx) == 10 and bits == sorted(bits, reverse=True)
    assert bits[0] == max(t.quant_bits for t in prose.tokens)
    for i in idx:
        top3 = prose.tokens[i].top
        assert len(top3) == 3
        assert all(isinstance(model.decode([tid]), str) and 0 <= tid < model.vocab_size for tid, _ in top3)
        probs = [p for _, p in top3]
        assert probs == sorted(probs, reverse=True) and sum(probs) <= 1.0 + 1e-9


def test_autopilot_runs(prose):
    runs = autopilot_runs(prose)
    assert [n for _, n in runs] == sorted((n for _, n in runs), reverse=True)
    for start, n in runs:
        assert all(t.quant_bits < 0.5 for t in prose.tokens[start : start + n])


def test_max_tokens_truncation(model):
    a = analyze(PROSE, model, max_tokens=30)
    assert a.truncated and len(a.tokens) == 30 and a.total_tokens > 30


# --- CLI ----------------------------------------------------------------------------


@pytest.mark.slow
def test_cli_no_color_json_and_truncation(tmp_path):
    out = tmp_path / "inspect.json"
    r = run_cli("inspect", SAMPLES / "prose.txt", "--no-color", "--json", out, "--max-tokens", "50", "--top", "5")
    assert "Top 5 most surprising tokens" in r.stdout
    assert "truncated to the first 50 of" in r.stdout
    assert "\x1b[" not in r.stdout  # plain text
    data = json.loads(out.read_text())
    assert len(data["tokens"]) == data["summary"]["tokens"] == 50
    assert data["summary"]["truncated"] is True
    assert {"id", "text", "ideal_bits", "quantized_bits", "block"} <= set(data["tokens"][0])


def test_render_keeps_bracketed_context(model):
    """Rich must not swallow '[import]' as markup in the tables."""
    import io

    from rich.console import Console

    a = analyze("import re\nimport sys\nfrom collections import Counter\n", model)
    console = Console(width=200, no_color=True, record=True, file=io.StringIO())
    render(a, model, console, 5, heatmap=False)
    out = console.export_text()
    assert "[import]" in out or "[from]" in out


def test_cli_non_utf8_exits_1(tmp_path):
    f = tmp_path / "bin.dat"
    f.write_bytes(b"\xff\xfe\x00binary")
    r = run_cli("inspect", f, check=False)
    assert r.returncode == 1
    assert r.stderr.startswith("llmzip: error:") and "UTF-8" in r.stderr and "Traceback" not in r.stderr


# --- LLM token budget (compress) --------------------------------------------------------


def test_llm_budget_skips_auto_but_not_forced(model):
    data = PROSE.encode()
    blob, info = api.compress_bytes(data, model=model, max_llm_tokens=10)
    assert info.llm_over_budget == (info.n_tokens, 10)
    assert info.mode in (ct.ZSTD, ct.STORED) and "LLM" not in info.candidates
    assert api.decompress_bytes(blob, model=model)[0] == data
    forced, finfo = api.compress_bytes(data, mode="llm", model=model, max_llm_tokens=10)
    assert finfo.mode == ct.LLM and finfo.llm_over_budget is None
    unlimited, uinfo = api.compress_bytes(data, model=model, max_llm_tokens=0)
    assert uinfo.mode == ct.LLM and unlimited == forced


def test_cli_budget_message(tmp_path):
    src = tmp_path / "p.txt"
    src.write_bytes(PROSE.encode())
    r = run_cli("compress", src, "--max-llm-tokens", "10")
    assert "LLM skipped:" in r.stdout and "> budget 10" in r.stdout and "--max-llm-tokens 0" in r.stdout
