"""Headless tests for `llmzip tui` (Textual App.run_test / Pilot).

Async bodies run via asyncio.run() inside plain sync tests (no extra pytest plugin).
One model instance is shared and injected into every app.
"""

import asyncio
import shutil
import time

import pytest
from textual.widgets import DataTable, Select

from llmzip import api
from llmzip import container as ct
from llmzip.inspect_cmd import analyze, top_surprising
from llmzip.model import NextTokenModel
from llmzip.tui.app import LlmzipApp
from llmzip.tui.heatmap import HeatmapView, layout
from llmzip.tui.screens import ConfirmScreen

from helpers import PARAGRAPH, SAMPLES, run_cli

SIZE = (140, 45)
OUR_GROUPS = {"job", "info", "model"}  # DirectoryTree's own loader worker never finishes


@pytest.fixture(scope="module")
def model():
    return NextTokenModel()


def copy_sample(tmp_path, name):
    dst = tmp_path / name
    shutil.copy(SAMPLES / name, dst)
    return dst


async def wait_idle(app, pilot, timeout=120.0):
    deadline = time.monotonic() + timeout
    await pilot.pause()
    while time.monotonic() < deadline:
        running = [w for w in app.workers if w.group in OUR_GROUPS and w.is_running]
        if not running and not app.busy:
            await pilot.pause()
            return
        await pilot.pause(0.05)
    raise AssertionError("app did not become idle")


def run(coro):
    return asyncio.run(coro)


# --- launch -------------------------------------------------------------------------


def test_launch_on_folder_and_file(tmp_path, model):
    md = copy_sample(tmp_path, "sample.md")
    (tmp_path / "blob.bin").write_bytes(bytes(range(256)) * 4)

    async def body():
        app = LlmzipApp(tmp_path, model=model)
        async with app.run_test(size=SIZE) as pilot:
            await wait_idle(app, pilot)
            assert app.current is None and app.start_dir == tmp_path.resolve()
            app.load_file(tmp_path / "blob.bin")
            await wait_idle(app, pilot)
            assert app.file_kind == "binary" and app.file_size == 1024
            assert "binary" in app.overview_text and "1,024 B" in app.overview_text

        app = LlmzipApp(md, model=model)
        async with app.run_test(size=SIZE) as pilot:
            await wait_idle(app, pilot)
            assert app.file_kind == "text" and app.file_size == md.stat().st_size
            assert "UTF-8 text" in app.overview_text and "tokens      196" in app.overview_text

    run(body())


# --- compress / decompress ------------------------------------------------------------


def test_compress_then_decompress_roundtrip(tmp_path, model):
    md = copy_sample(tmp_path, "sample.md")
    original = md.read_bytes()

    async def body():
        app = LlmzipApp(md, model=model)
        async with app.run_test(size=SIZE) as pilot:
            await wait_idle(app, pilot)
            await pilot.press("ctrl+o")
            await wait_idle(app, pilot)
            lz = tmp_path / "sample.md.lz"
            assert lz.exists() and "Compressed" in app.last_result
            md.rename(tmp_path / "original.md")
            app.load_file(lz)
            await wait_idle(app, pilot)
            assert app.file_kind == "lz" and "match: yes" in app.overview_text
            await pilot.press("ctrl+d")
            await wait_idle(app, pilot)
            assert "Decompressed" in app.last_result
        assert (tmp_path / "sample.md").read_bytes() == original

    run(body())


def test_tui_output_identical_to_cli(tmp_path, model):
    md = copy_sample(tmp_path, "sample.md")
    cli_out = tmp_path / "cli.lz"
    run_cli("compress", md, "-o", cli_out, "--mode", "llm", "--quiet")

    async def body():
        app = LlmzipApp(md, model=model)
        async with app.run_test(size=SIZE) as pilot:
            await wait_idle(app, pilot)
            app.query_one("#mode", Select).value = "llm"
            await pilot.pause()
            await pilot.press("ctrl+o")
            await wait_idle(app, pilot)

    run(body())
    assert (tmp_path / "sample.md.lz").read_bytes() == cli_out.read_bytes()


def test_overwrite_declined_leaves_file(tmp_path, model):
    md = copy_sample(tmp_path, "sample.md")
    lz = tmp_path / "sample.md.lz"
    lz.write_bytes(b"precious")

    async def body():
        app = LlmzipApp(md, model=model)
        async with app.run_test(size=SIZE) as pilot:
            await wait_idle(app, pilot)
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("n")
            await wait_idle(app, pilot)
            assert not isinstance(app.screen, ConfirmScreen)
            assert "Not overwriting" in app.last_notice

    run(body())
    assert lz.read_bytes() == b"precious"


def test_cancel_compress_writes_nothing(tmp_path, model):
    src = copy_sample(tmp_path, "prose_long.txt")  # 894 tokens, several seconds

    async def body():
        app = LlmzipApp(src, model=model)
        async with app.run_test(size=SIZE) as pilot:
            await wait_idle(app, pilot)
            await pilot.press("ctrl+o")
            deadline = time.monotonic() + 30
            while app.progress_done == 0 and time.monotonic() < deadline:
                await pilot.pause(0.02)
            assert app.busy and app.progress_done > 0
            await pilot.press("c")
            await wait_idle(app, pilot)
            assert "cancelled" in app.last_notice
            assert app.is_running

    run(body())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["prose_long.txt"]  # no .lz, no .tmp


def test_corrupt_lz_shows_error_and_keeps_running(tmp_path, model):
    blob, _ = api.compress_bytes(PARAGRAPH.encode(), mode="zstd")
    bad = bytearray(blob)
    bad[-3] ^= 0x01
    lz = tmp_path / "bad.txt.lz"
    lz.write_bytes(bytes(bad))

    async def body():
        app = LlmzipApp(lz, model=model)
        async with app.run_test(size=SIZE) as pilot:
            await wait_idle(app, pilot)
            await pilot.press("ctrl+d")
            await wait_idle(app, pilot)
            assert "Decompress failed" in app.last_error
            assert app.is_running and not app.busy

    run(body())
    assert not (tmp_path / "bad.txt").exists()


# --- heatmap and stats ----------------------------------------------------------------


def test_heatmap_cursor_updates_panel_and_matches_inspect(tmp_path, model):
    md = copy_sample(tmp_path, "sample.md")
    reference = analyze(md.read_text(encoding="utf-8"), model, top_k=5)

    async def body():
        app = LlmzipApp(md, model=model)
        async with app.run_test(size=SIZE) as pilot:
            await wait_idle(app, pilot)
            await pilot.press("a")
            await wait_idle(app, pilot)
            assert app.analysis is not None and app.selected_index == 0
            await pilot.press("2")  # opening the tab must focus the heatmap (no manual focus)
            await pilot.pause()
            view = app.query_one(HeatmapView)
            assert app.focused is view
            await pilot.press("right", "right", "right")
            await pilot.pause()
            assert view.cursor == app.selected_index == 3
            got, want = app.analysis.tokens[3], reference.tokens[3]
            assert (got.id, got.text, got.block, got.top) == (want.id, want.text, want.block, want.top)
            assert got.quant_bits == want.quant_bits and got.ideal_bits == want.ideal_bits
            panel = str(app.query_one("#token-panel").render())
            assert f"{want.quant_bits:.2f}" in panel and f"{want.prob:.3g}" in panel
            before = view.cursor
            await pilot.press("down")
            await pilot.pause()
            assert view.position_of(view.cursor)[0] > view.position_of(before)[0]
            assert app.selected_index == view.cursor
            await pilot.press("left")
            await pilot.pause()
            assert app.selected_index == view.cursor

    run(body())


def test_click_on_wrapped_line_selects_token(tmp_path, model):
    f = tmp_path / "long.txt"
    f.write_text(PARAGRAPH, encoding="utf-8")  # one long line: soft-wraps several times

    async def body():
        app = LlmzipApp(f, model=model)
        async with app.run_test(size=(100, 40)) as pilot:
            await wait_idle(app, pilot)
            await pilot.press("a")
            await wait_idle(app, pilot)
            await pilot.press("2")
            await pilot.pause()
            view = app.query_one(HeatmapView)
            lines = view.lines
            assert "\n" not in PARAGRAPH and len(lines) >= 3  # wrapped
            assert max(s.end for line in lines for s in line) <= view.scrollable_content_region.width
            span = lines[2][len(lines[2]) // 2]  # a token in the middle of the 3rd visual line
            await pilot.click(HeatmapView, offset=((span.start + span.end) // 2, 2))
            await pilot.pause()
            assert view.cursor == span.token == app.selected_index
            # Re-layout on resize: wider terminal -> fewer, wider visual lines.
            n_before = len(view.lines)
            await pilot.resize_terminal(160, 40)
            await pilot.pause()
            width = view.scrollable_content_region.width
            assert len(view.lines) < n_before
            assert max(s.end for line in view.lines for s in line) <= width
            assert view.cursor == span.token  # selection survives re-layout

    run(body())


def test_stats_row_jumps_to_heatmap_token(tmp_path, model):
    md = copy_sample(tmp_path, "sample.md")

    async def body():
        app = LlmzipApp(md, model=model)
        async with app.run_test(size=SIZE) as pilot:
            await wait_idle(app, pilot)
            await pilot.press("3")  # opening Stats starts the analysis
            await wait_idle(app, pilot)
            assert app.analysis is not None
            table = app.query_one("#top-table", DataTable)
            table.focus()
            await pilot.press("down", "down", "enter")
            await pilot.pause()
            expected = top_surprising(app.analysis, 15)[2]
            assert app.query_one(HeatmapView).cursor == expected == app.selected_index
            assert app.query_one("#tabs").active == "heatmap"

    run(body())


# --- layout unit tests ----------------------------------------------------------------


def test_layout_wraps_and_keeps_wide_tokens_selectable():
    texts = ["aaaa", "bbbbbbbbbbbbb", "\n", "", "🍕", "c\td"]
    lines = layout(texts, 6)
    for line in lines:
        assert sum(s.end - s.start for s in line) <= 6
        assert all(a.end <= b.start for a, b in zip(line, line[1:]))
    tokens = {s.token for line in lines for s in line}
    assert tokens == set(range(len(texts)))  # every token visible, even "" and "\n"
    wide = [s for line in lines for s in line if s.token == 1]
    assert len(wide) >= 2 and "".join(s.text for s in wide) == "bbbbbbbbbbbbb"
