"""`llmzip tui`: a Textual front end. All real work goes through api / inspect_cmd / bench.

Threading model: every slow task runs in a thread worker (group "job", one at a time;
the model loads once in its own worker). Workers touch the UI only through
`call_from_thread`. Cancel sets an Event; the progress callback handed to the api
raises `Cancelled` on the next token. Output is built in memory and written (atomically)
only after a job finishes, so a cancelled or failed job never writes anything.
"""

import math
import os
import threading
import time
import traceback
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Label,
    ProgressBar,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from llmzip import api, bench
from llmzip import container as ct
from llmzip import inspect_cmd
from llmzip.config import MAX_BLOCK_TOKENS
from llmzip.errors import Cancelled, LlmzipError, UsageError
from llmzip.fileops import (
    NOT_PRESENT,
    default_compress_output,
    default_decompress_output,
    identity_check,
    restored_output,
    write_atomic,
)
from llmzip.tui.heatmap import HeatmapView
from llmzip.tui.screens import ConfirmScreen, HelpScreen

TABS = ["overview", "heatmap", "stats", "compare"]
MODES = [("auto", "auto"), ("llm", "llm"), ("zstd", "zstd"), ("store", "store")]


class FileTree(DirectoryTree):
    """DirectoryTree without dotfiles (.DS_Store, .git, …)."""

    def filter_paths(self, paths):
        return [p for p in paths if not p.name.startswith(".")]


def _human(n: int) -> str:
    return f"{n:,} B" if n < 10_000 else f"{n:,} B ({n / 1024:.1f} KiB)"


class LlmzipApp(App):
    TITLE = "llmzip"
    CSS = """
    #tree { width: 36; border-right: solid $panel; }
    #restore-row { height: auto; margin-bottom: 1; }
    #restore-row Label { padding: 1 1; }
    #main { width: 1fr; }
    #actions { height: auto; margin: 1 0; }
    #actions Button { margin-right: 1; }
    #mode { width: 16; margin-right: 1; }
    #progress-row { height: auto; }
    #progress-label { margin-left: 2; width: 1fr; }
    #result { margin-top: 1; padding: 0 1; border: round $panel; height: auto; }
    #file-info { padding: 0 1; height: auto; }
    #heat-row { height: 1fr; }
    HeatmapView { width: 1fr; }
    #token-panel { width: 34; padding: 0 1; border-left: solid $panel; }
    #legend, #heat-notice { height: auto; padding: 0 1; }
    #top-table, #runs-table { height: auto; max-height: 16; margin-bottom: 1; }
    #bench-table { height: auto; margin-top: 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("question_mark", "help", "Help"),
        Binding("1", "show_tab('overview')", "Overview"),
        Binding("2", "show_tab('heatmap')", "Heatmap"),
        Binding("3", "show_tab('stats')", "Stats"),
        Binding("4", "show_tab('compare')", "Compare"),
        Binding("ctrl+o", "compress", "Compress"),
        Binding("ctrl+d", "decompress", "Decompress"),
        Binding("v", "verify", "Verify"),
        Binding("a", "analyze", "Analyze"),
        Binding("b", "bench", "Bench", show=False),
        Binding("c", "cancel", "Cancel"),
        Binding("t", "toggle_tree", "Tree"),
    ]

    busy: reactive[bool] = reactive(False)

    def __init__(self, path: str | Path = ".", model=None, max_tokens: int = 5000,
                 tab: str = "overview") -> None:
        super().__init__()
        if tab not in TABS:
            raise ValueError(f"tab must be one of {TABS}")
        self.start_tab = tab
        path = Path(path).expanduser().resolve()
        self.start_dir = path if path.is_dir() else path.parent
        self.start_file = path if path.is_file() else None
        self.max_tokens = max_tokens or None
        self.model = model
        self.model_ready = threading.Event()
        if model is not None:
            self.model_ready.set()
        self._cancel = threading.Event()
        # Current file and derived state (also read by tests).
        self.current: Path | None = None
        self.file_kind: str | None = None  # "text" | "binary" | "lz"
        self.file_size = 0
        self.text: str | None = None
        self.analysis: inspect_cmd.Analysis | None = None
        self.selected_index: int | None = None
        self.progress_done = 0
        self.last_result = ""
        self.last_error = ""
        self.last_notice = ""
        self.identity = None

    # --- layout -----------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield FileTree(str(self.start_dir), id="tree")
            with Vertical(id="main"):
                with TabbedContent(initial="overview", id="tabs"):
                    with TabPane("Overview [1]", id="overview"):
                        with VerticalScroll():
                            yield Static("Select a file in the tree.", id="file-info")
                            with Horizontal(id="actions"):
                                yield Select(MODES, value="auto", allow_blank=False, id="mode")
                                yield Button("Compress", id="compress", variant="primary")
                                yield Button("Decompress", id="decompress")
                                yield Button("Verify", id="verify")
                                yield Button("Cancel", id="cancel", variant="error", disabled=True)
                            with Horizontal(id="restore-row"):
                                yield Switch(value=True, id="restored")
                                yield Label("Restore as copy (.restored) + identity check")
                            with Horizontal(id="progress-row"):
                                yield ProgressBar(total=100, show_eta=False, id="progress")
                                yield Static("", id="progress-label")
                            yield Static("", id="result")
                    with TabPane("Heatmap [2]", id="heatmap"):
                        yield Static("Press a to analyze this file.", id="heat-notice")
                        with Horizontal(id="heat-row"):
                            yield HeatmapView(id="heatmap-view")
                            yield Static("", id="token-panel")
                        yield Static(self._legend(), id="legend")
                    with TabPane("Stats [3]", id="stats"):
                        with VerticalScroll():
                            yield Static("Press a to analyze this file.", id="stats-summary")
                            yield Static("", id="histogram")
                            yield Static(Text("Top surprising tokens (enter: show in heatmap)", style="bold"))
                            yield DataTable(id="top-table", cursor_type="row", zebra_stripes=True)
                            yield Static(Text("Autopilot stretches (each token < 0.5 bits)", style="bold"))
                            yield DataTable(id="runs-table", cursor_type="row", zebra_stripes=True)
                    with TabPane("Compare [4]", id="compare"):
                        with VerticalScroll():
                            yield Button("Run benchmark (b)", id="bench", variant="primary")
                            yield Static("llmzip (forced LLM and auto) vs gzip -9 / xz -9 / zstd -19 "
                                         "(piped, so no filename is stored).", id="bench-note")
                            yield DataTable(id="bench-table", zebra_stripes=True)
        yield Footer()

    def _legend(self) -> Text:
        legend = Text("quantized bits/token: ")
        for _, bg, fg, label in inspect_cmd.BUCKETS:
            legend.append(f" {label} ", style=f"{fg} on {bg}")
            legend.append(" ")
        return legend

    def on_mount(self) -> None:
        self.sub_title = str(self.start_dir)
        top = self.query_one("#top-table", DataTable)
        top.add_columns("#", "token", "bits", "prob", "context")
        runs = self.query_one("#runs-table", DataTable)
        runs.add_columns("start", "tokens", "bits", "text")
        self.query_one("#bench-table", DataTable).add_columns("method", "size", "bits/byte", "")
        self.query_one("#result", Static).display = False
        self._update_buttons()
        if not self.model_ready.is_set():
            self._set_label("Loading model…")
            self.load_model()
        if self.start_file:
            self.load_file(self.start_file)
        if self.start_tab != "overview":
            self.action_show_tab(self.start_tab)  # heatmap/stats start the analysis

    # --- model ------------------------------------------------------------------------

    @work(thread=True, group="model")
    def load_model(self) -> None:
        try:
            from llmzip.llmcodec import get_model

            self.model = get_model()
            self.model.fingerprint_bytes()
            self.model_ready.set()
            self.call_from_thread(self._model_loaded)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._error, f"could not load the model: {e}", e)

    def _model_loaded(self) -> None:
        if not self.busy:
            self._set_label("Model loaded.")
        if self.current:
            self.load_file(self.current)

    def _wait_model(self):
        if not self.model_ready.is_set():
            self.call_from_thread(self._set_label, "Loading model…")
            self.model_ready.wait()
        return self.model

    # --- file selection ---------------------------------------------------------------

    @on(DirectoryTree.FileSelected)
    def _file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.load_file(Path(event.path))

    def load_file(self, path: Path) -> None:
        path = Path(path)
        changed = path != self.current
        self.current = path
        self.sub_title = str(path)
        try:
            data = path.read_bytes()
        except OSError as e:
            self._error(f"cannot read {path}: {e}", e)
            return
        self.file_size = len(data)
        self.text = None
        if data[:4] == ct.MAGIC:
            self.file_kind = "lz"
        else:
            try:
                self.text = data.decode("utf-8")
                self.file_kind = "text"
            except UnicodeDecodeError:
                self.file_kind = "binary"
        if changed:
            self.analysis = None
            self.selected_index = None
            self.query_one(HeatmapView).clear()
            self.query_one("#heat-notice", Static).update("Press a to analyze this file.")
            self.query_one("#token-panel", Static).update("")
            self.query_one("#stats-summary", Static).update("Press a to analyze this file.")
            self.query_one("#histogram", Static).update("")
            for table in ("#top-table", "#runs-table", "#bench-table"):
                self.query_one(table, DataTable).clear()
            self.query_one("#result", Static).update("")
            self.query_one("#result", Static).display = False
        self._show_overview({})
        self.file_details(path, data)
        self._update_buttons()

    @work(thread=True, exclusive=True, group="info")
    def file_details(self, path: Path, data: bytes) -> None:
        """Token count (text) or header info (.lz); needs the tokenizer/model, so off-thread."""
        details: dict = {}
        try:
            if self.file_kind == "lz":
                c = ct.unpack(data)
                details["lz"] = c
                if c.mode == ct.LLM and self.model_ready.is_set():
                    details["fp_match"] = c.body.fingerprint == self.model.fingerprint_bytes()
            elif self.file_kind == "text" and self.text and self.model_ready.is_set():
                from llmzip.llmcodec import NotRoundTrippable, tokenize_checked

                try:
                    details["tokens"] = len(tokenize_checked(self.text, self.model))
                except NotRoundTrippable:
                    details["tokens"] = "not round-trippable (will use zstd)"
        except LlmzipError as e:
            details["lz_error"] = str(e)
        if path == self.current:
            self.call_from_thread(self._show_overview, details)

    def _show_overview(self, details: dict) -> None:
        if self.current is None:
            return
        kinds = {"text": "UTF-8 text", "binary": "binary", "lz": "llmzip .lz container"}
        t = Text()
        t.append(f"{self.current.name}\n", style="bold")
        t.append(f"size        {_human(self.file_size)}\n")
        t.append(f"type        {kinds[self.file_kind]}\n")
        if self.file_kind == "text":
            tokens = details.get("tokens", "… (loading model)" if not self.model_ready.is_set() else "…")
            t.append(f"tokens      {tokens}\n")
        if "lz_error" in details:
            t.append(f"header      INVALID: {details['lz_error']}\n", style="bold red")
        if "lz" in details:
            c = details["lz"]
            t.append(f"mode        {ct.MODE_NAMES[c.mode]}\n")
            t.append(f"original    {_human(c.original_size)}   crc32 {c.crc32:08x}\n")
            if c.mode == ct.LLM:
                b = c.body
                t.append(f"blocks      {len(b.blocks)} × ≤{b.block_size} tokens, {b.n_tokens} tokens\n")
                match = details.get("fp_match")
                fp = "… (loading model)" if match is None else ("yes" if match else "NO, decompression will be refused")
                t.append(f"fingerprint {b.fingerprint.hex()}  match: {fp}\n",
                         style="" if match is not False else "bold red")
        self.overview_text = t.plain
        self.query_one("#file-info", Static).update(t)

    # --- busy state, progress ---------------------------------------------------------

    def watch_busy(self, busy: bool) -> None:
        self._update_buttons()

    def _update_buttons(self) -> None:
        try:
            has = self.current is not None
            self.query_one("#compress", Button).disabled = self.busy or not has
            self.query_one("#decompress", Button).disabled = self.busy or self.file_kind != "lz"
            self.query_one("#verify", Button).disabled = self.busy or not has
            self.query_one("#bench", Button).disabled = self.busy or not has
            self.query_one("#cancel", Button).disabled = not self.busy
            self.query_one("#mode", Select).disabled = self.busy
            self.query_one("#restored", Switch).disabled = self.busy
        except Exception:  # noqa: BLE001  (widgets not mounted yet)
            pass

    def _set_label(self, text: str) -> None:
        self.query_one("#progress-label", Static).update(text)

    def _progress_factory(self, label: str):
        """api-style ProgressFactory that drives the ProgressBar and honours Cancel."""

        def factory(total: int):
            start = time.perf_counter()
            state = {"done": 0, "last": 0.0}
            self.call_from_thread(self._progress, label, 0, total, 0.0)

            def advance(n: int) -> None:
                if self._cancel.is_set():
                    raise Cancelled(f"{label} cancelled")
                state["done"] += n
                now = time.perf_counter()
                if now - state["last"] >= 0.1 or state["done"] >= total:
                    state["last"] = now
                    self.call_from_thread(self._progress, label, state["done"], total, now - start)

            return advance

        return factory

    def _progress(self, label: str, done: int, total: int, elapsed: float) -> None:
        self.progress_done = done
        self.query_one("#progress", ProgressBar).update(total=total, progress=done)
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else math.nan
        eta_s = "–" if math.isnan(eta) else f"{int(eta) // 60}:{int(eta) % 60:02d}"
        self._set_label(f"{label} {done}/{total} tok · {rate:.0f} tok/s · ETA {eta_s}")

    # --- job runner -------------------------------------------------------------------

    def _start(self, job, *args) -> None:
        if self.busy:
            self.notify("A job is already running (c to cancel).", severity="warning")
            return
        self.busy = True
        self._cancel.clear()
        self.progress_done = 0
        self.query_one("#progress", ProgressBar).update(total=None, progress=0)
        job(*args)

    def _job_wrapper(self, label: str, fn) -> None:
        """Run fn() in the worker thread with uniform cancel/error handling."""
        try:
            fn()
        except Cancelled:
            self.call_from_thread(self._notice, f"{label} cancelled — nothing written.")
        except (LlmzipError, UsageError, OSError, ValueError) as e:
            self.call_from_thread(self._error, f"{label} failed: {e}", e)
        except Exception as e:  # noqa: BLE001  (never crash the UI)
            self.call_from_thread(self._error, f"{label} failed unexpectedly: {e!r}", e)
        finally:
            self.call_from_thread(setattr, self, "busy", False)

    def _notice(self, message: str) -> None:
        self.last_notice = message
        self._set_label(message)
        self.notify(message)

    def _error(self, message: str, exc: BaseException | None = None) -> None:
        self.last_error = message
        self._set_label(message)
        self.notify(message, severity="error", timeout=10)
        if exc is not None and os.environ.get("LLMZIP_DEBUG") == "1":
            from llmzip.model import cache_dir

            try:
                log = cache_dir() / "tui-error.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a") as f:
                    f.write(f"--- {time.ctime()}: {message}\n")
                    f.write("".join(traceback.format_exception(exc)))
            except OSError:
                pass

    def _result(self, text: Text | str) -> None:
        self.last_result = text.plain if isinstance(text, Text) else text
        result = self.query_one("#result", Static)
        result.update(text)
        result.display = True

    def _confirm_then(self, out: Path, start) -> None:
        if out.exists():
            def answer(yes: bool | None) -> None:
                if yes:
                    start()
                else:
                    self._notice(f"Not overwriting {out.name}.")
            self.push_screen(ConfirmScreen(f"{out} already exists.\nOverwrite it?"), answer)
        else:
            start()

    # --- compress / decompress / verify -----------------------------------------------

    def action_compress(self) -> None:
        if self.busy or self.current is None:
            return
        src = self.current
        out = default_compress_output(src)
        mode = self.query_one("#mode", Select).value
        self._confirm_then(out, lambda: self._start(self.run_compress, src, out, mode))

    @work(thread=True, exclusive=True, group="job")
    def run_compress(self, src: Path, out: Path, mode: str) -> None:
        def job():
            data = src.read_bytes()
            needs_model = mode in ("auto", "llm") and data[:4] != ct.MAGIC
            model = self._wait_model() if needs_model else None
            t0 = time.perf_counter()
            blob, info = api.compress_bytes(data, mode, MAX_BLOCK_TOKENS,
                                            self._progress_factory("Compressing"), model=model)
            if self._cancel.is_set():
                raise Cancelled("compress cancelled")
            write_atomic(out, blob)
            elapsed = time.perf_counter() - t0
            self.call_from_thread(self._compress_done, src, out, data, blob, info, elapsed)

        self._job_wrapper("Compress", job)

    def _compress_done(self, src, out, data, blob, info, elapsed) -> None:
        t = Text()
        t.append(f"Compressed {src.name} → {out.name}\n", style="bold green")
        t.append(f"mode {ct.MODE_NAMES[info.mode]}   {len(data):,} → {len(blob):,} B   "
                 f"ratio {len(data) / len(blob):.2f}×")
        if info.mode == ct.LLM and info.chars:
            t.append(f"   {len(blob) * 8 / info.chars:.3f} bits/char")
        t.append(f"   {elapsed:.1f} s\n")
        cands = ", ".join(f"{k} {v:,} B" for k, v in info.candidates.items())
        t.append(f"candidates: {cands}\n", style="dim")
        for note in api.skip_notes(info):
            t.append(note + "\n", style="yellow")
        self._result(t)
        self._set_label(f"Wrote {out.name}")
        self.notify(f"Wrote {out.name}")
        self._refresh_tree()

    def action_decompress(self) -> None:
        if self.busy or self.current is None:
            return
        if self.file_kind != "lz":
            self.notify("Decompress needs a .lz file.", severity="warning")
            return
        src = self.current
        try:
            restored = self.query_one("#restored", Switch).value
            out = restored_output(src) if restored else default_decompress_output(src)
        except UsageError as e:
            self._error(str(e))
            return
        self._confirm_then(out, lambda: self._start(self.run_decompress, src, out))

    @work(thread=True, exclusive=True, group="job")
    def run_decompress(self, src: Path, out: Path) -> None:
        def job():
            blob = src.read_bytes()
            model = self._wait_model() if ct.unpack(blob).mode == ct.LLM else None
            t0 = time.perf_counter()
            data, c = api.decompress_bytes(blob, self._progress_factory("Decompressing"), model=model)
            if self._cancel.is_set():
                raise Cancelled("decompress cancelled")
            write_atomic(out, data)
            elapsed = time.perf_counter() - t0
            ident = identity_check(src, out) if out != default_decompress_output(src) else None
            self.call_from_thread(self._decompress_done, src, out, blob, data, c, elapsed, ident)

        self._job_wrapper("Decompress", job)

    def _decompress_done(self, src, out, blob, data, c, elapsed, ident=None) -> None:
        t = Text()
        t.append(f"Decompressed {src.name} → {out.name}\n", style="bold green")
        t.append(f"mode {ct.MODE_NAMES[c.mode]}   {len(blob):,} → {len(data):,} B   CRC32 ok   {elapsed:.1f} s\n")
        if out != default_decompress_output(src):  # a restored copy: compare with the original
            if ident is None:
                t.append(NOT_PRESENT + "\n", style="dim")
            else:
                t.append(ident.line() + "\n", style="bold green" if ident.identical else "bold red")
        self.identity = ident
        self._result(t)
        self._set_label(f"Wrote {out.name}")
        self.notify(f"Wrote {out.name}")
        self._refresh_tree()

    def action_verify(self) -> None:
        if self.busy or self.current is None:
            return
        self._start(self.run_verify, self.current, self.query_one("#mode", Select).value)

    @work(thread=True, exclusive=True, group="job")
    def run_verify(self, src: Path, mode: str) -> None:
        def job():
            data = src.read_bytes()
            if data[:4] == ct.MAGIC:
                model = self._wait_model() if ct.unpack(data).mode == ct.LLM else None
                restored, c = api.decompress_bytes(data, self._progress_factory("Verifying"), model=model)
                msg = f"Verified {src.name}: decodes to {len(restored):,} B, size and CRC32 ok. Nothing written."
            else:
                model = self._wait_model() if mode in ("auto", "llm") else None
                blob, info = api.compress_bytes(data, mode, MAX_BLOCK_TOKENS,
                                                self._progress_factory("Verify: compressing"), model=model)
                restored, _ = api.decompress_bytes(blob, self._progress_factory("Verify: decompressing"),
                                                   model=model)
                if restored != data:
                    raise LlmzipError("round trip produced different bytes")
                msg = (f"Verified {src.name}: {ct.MODE_NAMES[info.mode]} round trip is byte-identical "
                       f"({len(data):,} → {len(blob):,} B). Nothing written.")
            self.call_from_thread(self._verify_done, msg)

        self._job_wrapper("Verify", job)

    def _verify_done(self, msg: str) -> None:
        self._result(Text(msg, style="bold green"))
        self._notice(msg)

    def _refresh_tree(self) -> None:
        try:
            self.query_one(DirectoryTree).reload()
        except Exception:  # noqa: BLE001
            pass

    def action_cancel(self) -> None:
        if self.busy:
            self._cancel.set()
            self._set_label("Cancelling…")

    @on(Button.Pressed)
    def _button(self, event: Button.Pressed) -> None:
        actions = {"compress": self.action_compress, "decompress": self.action_decompress,
                   "verify": self.action_verify, "cancel": self.action_cancel, "bench": self.action_bench}
        if event.button.id in actions:
            actions[event.button.id]()

    # --- analysis (heatmap + stats) -----------------------------------------------------

    @on(TabbedContent.TabActivated)
    def _tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "heatmap":
            self.query_one(HeatmapView).focus()  # arrow keys move the token cursor right away
        if event.pane.id in ("heatmap", "stats") and self.analysis is None and not self.busy \
                and self.file_kind == "text":
            self.action_analyze()

    def action_show_tab(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab

    def action_analyze(self) -> None:
        if self.busy or self.current is None:
            return
        if self.file_kind != "text" or not self.text:
            self.notify("The heatmap needs a non-empty UTF-8 text file.", severity="warning")
            return
        self.query_one("#heat-notice", Static).update("Analyzing…")
        self._start(self.run_analyze, self.current, self.text)

    @work(thread=True, exclusive=True, group="job")
    def run_analyze(self, path: Path, text: str) -> None:
        def job():
            model = self._wait_model()
            a = inspect_cmd.analyze(text, model, MAX_BLOCK_TOKENS, self.max_tokens,
                                    self._progress_factory("Analyzing"), top_k=5)
            top = [model.decode([tid]) for t in a.tokens for tid, _ in t.top]  # decode off-thread
            self.call_from_thread(self._show_analysis, path, a, top)

        self._job_wrapper("Analyze", job)

    def _show_analysis(self, path: Path, a: inspect_cmd.Analysis, top_texts: list[str]) -> None:
        if path != self.current:
            return
        self.analysis = a
        k = len(a.tokens[0].top) if a.tokens else 0
        self._top_texts = [top_texts[i * k : (i + 1) * k] for i in range(len(a.tokens))]
        s = inspect_cmd.summary(a)
        notice = Text(f"{s['tokens']} tokens, {s['bits_per_char']:.3f} bits/char. ")
        if a.truncated:
            notice.append(f"Truncated to the first {len(a.tokens)} of {a.total_tokens} tokens (--max-tokens).",
                          style="bold yellow")
        self.query_one("#heat-notice", Static).update(notice)
        view = self.query_one(HeatmapView)
        view.set_tokens([t.text for t in a.tokens], [t.quant_bits for t in a.tokens])
        if self.query_one("#tabs", TabbedContent).active == "heatmap":
            view.focus()

        summary = Text()
        summary.append(f"{path.name}\n", style="bold")
        summary.append(
            f"tokens {s['tokens']}" + (f" (of {s['total_tokens']}, truncated)" if a.truncated else "")
            + f"   characters {s['characters']}   blocks {s['blocks']}\n"
            f"bits {s['quantized_bits']:.1f} (ideal {s['ideal_bits']:.1f})   bits/char {s['bits_per_char']:.3f}\n"
            f"predicted LLM payload {s['predicted_payload_bytes']:,} B   actual {s['actual_payload_bytes']:,} B\n"
            f"predictability {s['predictability_pct']:.1f}% of tokens cost < 1 bit\n"
        )
        self.query_one("#stats-summary", Static).update(summary)
        self.query_one("#histogram", Static).update(self._histogram(a))

        top = self.query_one("#top-table", DataTable)
        top.clear()
        self._top_rows = inspect_cmd.top_surprising(a, 15)
        for i in self._top_rows:
            t = a.tokens[i]
            top.add_row(str(i), Text(repr(t.text)), f"{t.quant_bits:.1f}", f"{t.prob:.2g}",
                        Text(inspect_cmd.context(a, i)))
        runs = self.query_one("#runs-table", DataTable)
        runs.clear()
        for start, n in inspect_cmd.autopilot_runs(a):
            seg = a.tokens[start : start + n]
            snippet = "".join(t.text for t in seg).replace("\n", "\\n")
            runs.add_row(str(start), str(n), f"{sum(t.quant_bits for t in seg):.1f}",
                         Text(snippet[:60] + ("…" if len(snippet) > 60 else "")))
        self._set_label("Analysis done.")

    def _histogram(self, a: inspect_cmd.Analysis) -> Text:
        counts = [0] * len(inspect_cmd.BUCKETS)
        for t in a.tokens:
            for i, (limit, *_rest) in enumerate(inspect_cmd.BUCKETS):
                if t.quant_bits < limit:
                    counts[i] += 1
                    break
        total = max(len(a.tokens), 1)
        peak = max(max(counts), 1)
        out = Text("Tokens per bit bucket\n", style="bold")
        for (limit, bg, fg, label), n in zip(inspect_cmd.BUCKETS, counts):
            out.append(f" {label:>7} ", style=f"{fg} on {bg}")
            out.append(f" {n:>5} ({100 * n / total:4.1f}%) ")
            out.append("█" * round(40 * n / peak) + "\n", style=bg)
        return out

    @on(HeatmapView.TokenSelected)
    def _token_selected(self, event: HeatmapView.TokenSelected) -> None:
        a = self.analysis
        if a is None or not 0 <= event.index < len(a.tokens):
            return
        self.selected_index = event.index
        t = a.tokens[event.index]
        panel = Text()
        panel.append(f"token #{event.index}\n", style="bold")
        panel.append(f"{t.text!r}\n\n")
        panel.append(f"bits (quantized) {t.quant_bits:.2f}\n")
        panel.append(f"bits (ideal)     {t.ideal_bits:.2f}\n")
        panel.append(f"probability      {t.prob:.3g}\n")
        panel.append(f"block            {t.block}\n")
        panel.append(f"token id         {t.id}\n\n")
        panel.append("model's top predictions\n", style="bold")
        for (tid, p), txt in zip(t.top, self._top_texts[event.index]):
            mark = "→" if tid == t.id else " "
            panel.append(f" {mark} {p:6.1%}  {txt!r}\n")
        self.query_one("#token-panel", Static).update(panel)

    @on(DataTable.RowSelected, "#top-table")
    def _top_row_selected(self, event: DataTable.RowSelected) -> None:
        if self.analysis is None:
            return
        index = self._top_rows[event.cursor_row]
        self.action_show_tab("heatmap")
        view = self.query_one(HeatmapView)
        view.select(index)
        view.focus()

    # --- compare ----------------------------------------------------------------------

    def action_bench(self) -> None:
        if self.busy or self.current is None:
            return
        self.action_show_tab("compare")
        self._start(self.run_bench, self.current)

    @work(thread=True, exclusive=True, group="job")
    def run_bench(self, path: Path) -> None:
        def job():
            model = self._wait_model()
            row = bench.bench_file(path, model, [MAX_BLOCK_TOKENS], decompress=False,
                                   progress=self._progress_factory("Benchmark: LLM"))
            self.call_from_thread(self._show_bench, row)

        self._job_wrapper("Benchmark", job)

    def _show_bench(self, row: dict) -> None:
        n = row["bytes"]
        entries = []
        if row["llm"]:
            entries.append(("llmzip --mode llm", row["llm"]["size"]))
        auto = row["auto"]
        entries.append((f"llmzip auto (picks {auto['mode']})", auto["size"]))
        for name in bench.EXTERNAL:
            entries.append((name, row[name]["size"]))
        best = min(size for _, size in entries)
        table = self.query_one("#bench-table", DataTable)
        table.clear()
        for name, size in entries:
            style = "bold green" if size == best else ""
            table.add_row(Text(name, style=style), Text(f"{size:,}", style=style),
                          Text(f"{size * 8 / max(n, 1):.3f}", style=style),
                          Text("◀ smallest" if size == best else "", style=style))
        if not row["llm"]:
            table.add_row("llmzip --mode llm", "n/a", "n/a", "not UTF-8 text")
        self._set_label(f"Benchmark done for {row['file']} ({n:,} B).")

    # --- misc -------------------------------------------------------------------------

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_toggle_tree(self) -> None:
        tree = self.query_one("#tree")
        tree.display = not tree.display


def run(path: str = ".", max_tokens: int = 5000, tab: str = "overview") -> None:
    LlmzipApp(path, max_tokens=max_tokens, tab=tab).run()
