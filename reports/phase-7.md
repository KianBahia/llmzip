# Phase 7 report: Textual terminal app (`llmzip tui`)

## 1. Summary
`llmzip tui [PATH]` is an interactive Textual app with:
- a file browser;
- an Overview tab: file and `.lz` info, compress, decompress and verify, a mode selector, live progress with tok/s and ETA, and cancel;
- a soft-wrapped, clickable token **Heatmap** with a details panel showing the model's top-5 guesses;
- **Stats**: summary, histogram, top-N table that jumps to the heatmap, and autopilot stretches;
- **Compare**: bench against gzip, xz and zstd.

It is a front end only: all work goes through `api`, `inspect_cmd` and `bench`. A `.lz` made by the TUI is byte-identical to the CLI's, compression output is byte-identical to Phase 6, and all 334 tests pass.

## 2. Environment
Unchanged except for the one approved addition, **textual 8.2.8**: the latest stable on PyPI, which also pulls in `mdit-py-plugins` and `platformdirs`. The rest is macOS 15.7.1, Apple M4 Pro, uv 0.13.1, Python 3.12.15, torch 2.14.1, transformers 5.20.1, numpy 2.5.4, zstandard 0.25.0 and rich 15.0.0.

## 3. Files created/changed
| File | Purpose |
|---|---|
| `src/llmzip/tui/__init__.py` (new) | `run(path, max_tokens)` |
| `src/llmzip/tui/app.py` (new) | `LlmzipApp`: layout, the four tabs, workers, progress and cancel, overwrite confirm, error handling, key bindings |
| `src/llmzip/tui/heatmap.py` (new) | `layout()` (soft-wrap into per-visual-line spans) and `HeatmapView(ScrollView)` (line API, cursor, ↑/↓ by visual line, exact click mapping, re-layout on resize) |
| `src/llmzip/tui/screens.py` (new) | `ConfirmScreen` (overwrite y/n) and `HelpScreen` (all keys) |
| `src/llmzip/fileops.py` (new) | `default_compress_output`, `default_decompress_output`, `check_output` and `write_atomic`, moved out of cli.py and shared by CLI and TUI |
| `src/llmzip/errors.py` | `UsageError` (moved here), `Cancelled(LlmzipError)` |
| `src/llmzip/api.py` | `skip_notes(info)`: the "LLM skipped: …" texts shared by CLI and TUI |
| `src/llmzip/inspect_cmd.py` | `analyze(..., top_k=3)`, with the field renamed `top3` → `top`. CLI and JSON still show and store 3 predictions under the same `"top3"` key |
| `src/llmzip/bench.py` | `bench_file(..., progress=None)` passes progress through to `compress_bytes` |
| `src/llmzip/cli.py` | Uses `fileops` and `skip_notes` with identical output; new `tui [PATH] [--max-tokens N]` subcommand |
| `tests/test_tui.py` (new) | 10 headless TUI tests |
| `tests/test_inspect.py` | `.top3` → `.top` |
| `pyproject.toml`, `uv.lock` | `textual>=8.2.8` |
| `README.md` | "Terminal app" section (launch, key table, walkthrough); Quickstart line |
| `CLAUDE.md` | TUI architecture notes; phase 7 in the roadmap; approved-dependency note (outside the verbatim HARD RULES) |
| `reports/phase-6.md` | Commit-hash edit carried over from Phase 6 |
| `reports/phase-7.md` | This report |

## 4. Key decisions
- **No duplicated logic.** The TUI calls:
  - `api.compress_bytes` and `api.decompress_bytes` for compress, decompress and verify;
  - `inspect_cmd.analyze(top_k=5)`, `summary`, `top_surprising`, `autopilot_runs`, `context` and `BUCKETS` for the heatmap and stats;
  - `bench.bench_file` and `bench.EXTERNAL` for Compare;
  - `fileops` and `api.skip_notes`, shared with the CLI.

  The only new logic is presentation.
- **Workers:**
  - The model loads once at startup in a background thread worker. The Overview says "Loading model…", and the token count and fingerprint match fill in when it's ready.
  - Jobs (compress, decompress, verify, analyze, bench) are thread workers in group `"job"`, one at a time. A `busy` reactive disables the buttons and the mode selector, and enables Cancel.
  - Workers touch the UI only via `call_from_thread`.
  - STORED and ZSTD jobs never wait for the model.
- **Cancellation:**
  - `c` (or the Cancel button) sets a `threading.Event`. The api-style progress callback checks it on every token and raises `errors.Cancelled`, which the job wrapper turns into a "… cancelled — nothing written" notice.
  - Every job builds its output in memory and calls `write_atomic` only after finishing, so there is never a partial or temp file to clean up.
  - Progress UI updates are throttled to about 10 per second.
- **Overwrite protection:** output names follow the CLI (`fileops`). If the target exists, `ConfirmScreen` asks; the focus starts on Cancel, `n`/`escape` decline, and declining writes nothing.
- **Errors:** `LlmzipError`, `UsageError`, `OSError` and `ValueError` in a job become an error notification and a status line, and anything else is caught too. With `LLMZIP_DEBUG=1`, tracebacks are appended to `cache_dir()/tui-error.log`. The app never exits on an error.
- **Heatmap:**
  - **Rendering:** Textual's line API with **soft-wrap to the view width**, re-laid out on resize. A token that would cross the edge moves to the next line; a token wider than the view is split visually but stays one selectable token.
  - **Spans:** `(start, end, token)` spans are recorded per *visual* line, so a click maps exactly to a token. ↑/↓ move to the token at the same column on the next or previous visual line.
  - **Display-only choices:** tabs are drawn as 4 spaces and `\r` isn't drawn. A token that would occupy no cells (a lone `\n`, or the first half of a split multi-byte character) gets one space cell, so every token can be seen and clicked.
  - **Scrollbar:** `scrollbar-gutter: stable` keeps the content width fixed when the scrollbar appears. This was a real bug caught by the click test: the layout was computed 2 columns too wide.
  - **Colours and focus:** the same 6 buckets and colours as `inspect`. Opening the tab focuses the heatmap, so the arrow keys work immediately; this was found from a screenshot, see section 8.
- **On-demand analysis:** it runs only when the Heatmap or Stats tab is opened without an analysis, or when `a` is pressed. Selecting a file never triggers it. `--max-tokens` (default 5000) gives a visible truncation notice.
- **Keys:**
  - `1`–`4` switch tabs, as you required in place of tab/shift+tab. Textual's tab and shift+tab focus navigation is untouched.
  - `q` quit, `?` help, `ctrl+o` compress, `ctrl+d` decompress, `v` verify, `a` analyze, `b` bench, `c` cancel.
  - All of them are in the Footer (bench is listed in help) and in the help screen.
- **Verify:** for a `.lz` file, a full in-memory decode with the size and CRC check. For other files, an in-memory compress → decompress → compare using the selected mode. Nothing is written in either case.

## 5. Commands run and results
| Command | Result | Key output |
|---|---|---|
| Phase 6 baseline: `compress --mode llm` and auto for `bench/samples/*.txt`, plus `sample.md --mode llm`, before any edit | pass | 2,196 / 199 / 526 / 154 B |
| `.gitignore` check: `od -c`, then `git check-ignore -v --no-index bench/results/dummy_20990101.json testcompression/dummy.txt` | pass | the two rules were already on separate lines (10 and 11); only the trailing newline was missing; both rules match |
| `uv add textual` | pass | textual 8.2.8 |
| `uv run pytest tests/test_inspect.py tests/test_cli.py` (after the refactors) | pass | 34 passed |
| `uv run llmzip --help` | pass | `{compress,decompress,bench,inspect,tui,info}` |
| Headless smoke run, injected model and background model load | pass | Overview shows `tokens 196` in both cases |
| `uv run pytest tests/test_tui.py` | 1 fail, then pass | wrap-width bug, fixed with `scrollbar-gutter: stable` → 10 passed |
| Headless screenshots (SVG → PNG via `qlmanage`), inspected visually | pass after fixes | found the heatmap focus issue, an animated idle bar, an empty result box and histogram scaling; all fixed and re-shot |
| `uv run pytest` | pass | 334 passed in 186.06s |
| Byte identity vs Phase 6 | pass | 7/7 identical (see section 6) |

## 6. Smoke test results

**Byte identity vs Phase 6** (baseline made from HEAD `361f86b` before any edit):

| file | `--mode llm` | auto |
|---|---|---|
| blocktest.txt (2,196 B) | ✅ identical | ✅ identical |
| prose.txt (199 B) | ✅ identical | ✅ identical |
| prose_long.txt (526 B) | ✅ identical | ✅ identical |
| sample.md (154 B, CLI) | ✅ identical | — |

**TUI vs CLI:** `test_tui_output_identical_to_cli` compresses `sample.md` with mode llm through the TUI's worker thread and compares it with `llmzip compress --mode llm` run in a subprocess. ✅ **Byte-identical.** Because the CLI output equals the Phase 6 baseline, the TUI's output does too.

**What I checked visually** (headless 150×42 screenshots of sample.py, converted to PNG and inspected):
- **Overview:** size 761 B, UTF-8 text, 309 tokens. After compress: "mode LLM 761 → 207 B ratio 3.68× 2.176 bits/char", plus the candidates line.
- **Heatmap:** the layout matches the source (indentation, blank lines), and line-start tokens show red. After 19 × → then ↓, the cursor sits on token #22 `'from'`: 24.00 quantized bits, 28.81 ideal, p 2.13e-09, with GPT-2's top 5 led by `'\n'` at 100%.
- **Stats:** summary (predicted 165 B = actual 165 B), a coloured histogram, the top-15 table and autopilot runs.
- **Compare:** llmzip llm 207 B and auto (LLM) 207 B are highlighted as smallest, against gzip 444, xz 540 and zstd 452.
- **Help:** the full key list.

## 7. Test results
`334 passed in 186.06s`: Phase 6's 324 plus 10 in `tests/test_tui.py` (about 23 s for those):
1. **Launch:** on a folder (no file selected; a binary file loads as "binary, 1,024 B") and on a file ("UTF-8 text", `tokens 196`).
2. **Round trip:** compress then decompress through `ctrl+o` and `ctrl+d`. The `.lz` overview shows "match: yes", and the restored file is byte-identical.
3. **TUI vs CLI:** the `.lz` is byte-identical to the CLI's (`--mode llm`).
4. **Overwrite protection:** `ConfirmScreen` appears, `n` declines, and the existing file still holds its original bytes.
5. **Cancel:** `prose_long.txt` is compressed and `c` pressed once progress is above 0. A "cancelled" notice appears, the directory contains only the input (no `.lz`, no `.tmp`), and the app is still running.
6. **Corrupt `.lz`:** decompress reports "Decompress failed …", the app keeps running and is no longer busy, and no output is created.
7. **Heatmap keyboard:**
   - opening the tab focuses the heatmap;
   - → → → gives cursor 3, and the panel's bits and probability match the token;
   - the token's id, text, block, top-5, ideal and quantized bits equal `inspect_cmd.analyze(top_k=5)`;
   - ↓ moves to a later visual line, and ← keeps the panel in sync.
8. **Click on a wrapped line:** a click on a token on the 3rd visual line of a single-line paragraph selects it. Every span stays within the view width. After a resize to 160 columns there are fewer lines, they still fit, and the selection is kept.
9. **Stats:** opening Stats auto-analyzes. ↓ ↓ enter on the top-N table puts the heatmap cursor on `top_surprising(...)[2]` and switches to the Heatmap tab.
10. **Layout unit test:** wrapping, a wide token split but selectable, and every token (including `""` and `"\n"`) visible.

## 8. Deviations from the plan
- **Tabs and wrapping:** as you changed in review: `1`–`4` for tabs instead of tab/shift+tab, and soft-wrap instead of horizontal scroll.
- **`.gitignore`:** your uncommitted edit had only removed the trailing newline. Adding it back makes the file byte-identical to your commit `361f86b`, so **no `.gitignore` change is in this commit**. The rules are on separate lines, and `git check-ignore -v` confirms both.
- **Fixes found by inspecting headless screenshots** (I can't use a real TTY):
  - **The heatmap didn't take focus,** so the arrow keys did nothing until you clicked it. The tab now focuses it, and the test asserts this without focusing the view manually.
  - **The idle progress bar showed an indeterminate animation,** which looked like work was in progress. It now shows 0%.
  - **The empty result box** is now hidden until there is a result.
  - **Histogram bars** are now scaled to the largest bucket and coloured.
- **Panel widths:** the side panels are narrower (tree 28, token panel 34 columns), so the heatmap has room at 100 columns.
- **Extras:**
  - `v` (verify) and `b` (bench) keys;
  - a Cancel button;
  - the candidates line in the result;
  - `llmzip tui --max-tokens`.
- **Test plumbing:** `workers.wait_for_complete()` hangs, because `DirectoryTree`'s internal `_loader` worker never finishes. The tests therefore wait on llmzip's own worker groups.

## 9. Problems and open questions
- **No unresolved failures.**
- **Look and feel is not human-verified:** I checked headless screenshots, not a real terminal. Please run the manual script below.
- **Cancel granularity:** cancel acts on the next token. That is effectively instant for LLM work, but tokenizing a very large file, or a zstd-19 run on a large binary, can't be interrupted mid-way.
- **Speed:** the heatmap and analysis take as long as compression (about 100–150 tok/s), so 5,000 tokens takes about 40–50 s with a progress bar.
- **Narrow terminals:** below about 100 columns the heatmap gets narrow. The tree and token panel widths are fixed.
- **Decoding in the panel:** the token panel decodes top-5 predictions as single tokens, so a prediction that is part of a multi-byte character shows `�`.

## 10. Notes for later phases
- Possible TUI follow-ups:
  - a toggle to hide the file tree on narrow screens;
  - an option to compress to a different folder;
  - "open in heatmap" for `.lz` files (decode, then analyze);
  - showing the skip heuristic's estimate live.
- The project-level future ideas in CLAUDE.md are unchanged: parallel block coding, a leaner bit-identical forward pass, sliding-window format v2, a fine-tuned model and portability.

## 11. Git
- Commit: `82a37aa`
- Message: `Phase 7: Textual terminal app`
- I staged the files explicitly, and this commit includes the Phase 6 report hash edit. It has not been pushed.
- The hash was written into this file after the commit, so this line is an uncommitted edit.

## Manual test script (for your terminal)
1. `uv run llmzip tui bench/samples`
   - **Expect:** the file tree on the left, the Overview tab, "Loading model…", then "Model loaded." in a few seconds.
   - **Check:** the Footer lists q ? 1 2 3 4 ^o ^d v a c, and `?` opens the help screen (escape closes it).
2. Select **sample.py** in the tree.
   - **Expect:** size 761 B, UTF-8 text, 309 tokens.
   - Press **ctrl+o**: the progress bar fills with tok/s and ETA, and the result reads about "mode LLM 761 → 207 B ratio 3.68×".
   - `sample.py.lz` appears in the tree.
3. Press **ctrl+o** again.
   - **Expect:** an "already exists, overwrite?" dialog. Press **n**: nothing changes.
   - Select `sample.py.lz`: the header shows mode LLM, 1 block and "match: yes".
   - Press **ctrl+d**, answer **y** to overwrite `sample.py`, and it decompresses with "CRC32 ok".
4. Press **2** (Heatmap).
   - **Expect:** an analysis progress bar, then the code coloured by surprise, with line-start indentation in red. The token panel shows token #0, and the legend sits at the bottom.
   - Use the arrow keys and click a few tokens: the panel updates (bits, probability, top-5).
   - Resize the terminal: the text re-wraps and the colours stay aligned.
   - Check readability in both a dark and a light terminal theme.
5. Press **3** (Stats).
   - **Expect:** the summary with predicted = actual payload, a coloured histogram, and the top-surprising table.
   - Focus the table (tab or click), pick a row and press **enter**: the Heatmap tab opens with that token selected.
6. Press **4**, then **b** (Compare).
   - **Expect:** a table with llmzip at about 207 B highlighted in green as smallest, and gzip, xz and zstd at 444–540 B.
7. **Cancel:** select **reports_concat.md** and press **ctrl+o**. While the bar is moving, press **c**.
   - **Expect:** "Compress cancelled — nothing written", and no `reports_concat.md.lz` in the tree.
   - Then press **q** to quit. Clean up the `sample.py.lz` from step 2 if you like.
