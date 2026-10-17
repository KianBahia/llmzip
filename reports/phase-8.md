# Phase 8 report: restored copies, TUI polish, Finder actions

## 1. Summary
- **(A) Restored copies:** decompress can write a restored *copy* (`report.md.lz` → `report.restored.md`) and prove it matches the original: `Original vs restored: IDENTICAL ✓` with sizes and SHA-256s, or `DIFFERENT ✗`. This works in the CLI (`--restored`, plus `--unique` numbering) and in the TUI (a "Restore as copy" switch, on by default).
- **(B) TUI polish:** dotfiles are hidden, `t` toggles the file tree, the tree is wider, and `--tab` opens directly on a tab.
- **(C) Finder Quick Actions:** three right-click actions, as zsh scripts plus a generator for the `.workflow` bundles. I tested the scripts directly and the bundles only in a temp folder; **install.sh was not run against ~/Library.**

Compression output is byte-identical to Phase 7, there are no new Python dependencies, and all 368 tests pass.

## 2. Environment
Unchanged: macOS 15.7.1, Apple M4 Pro, uv 0.13.1 (`/opt/homebrew/bin/uv`), Python 3.12.15, torch 2.14.1, transformers 5.20.1, numpy 2.5.4, textual 8.2.8, zstandard 0.25.0, rich 15.0.0, Automator 2.10 (build 528), Run Shell Script action 2.0.3. No new dependencies.

## 3. Files created/changed
| File | Purpose |
|---|---|
| `src/llmzip/fileops.py` | `restored_output`, `unique_path(where="last"\|"base")`, `identity_check` → `Identity` (`.identical`, `.line()`), `NOT_PRESENT` |
| `src/llmzip/cli.py` | `decompress --restored` and `--unique` (prints the identity line); `compress --unique`; `tui --tab` |
| `src/llmzip/tui/app.py` | "Restore as copy (.restored) + identity check" `Switch` (on by default); identity line in green or red in the result panel; `FileTree` hides dotfiles; `t` toggles the tree; tree width 28 → 36; `tab=` start option; short "Wrote NAME" status |
| `src/llmzip/tui/__init__.py`, `tui/screens.py` | `tab` pass-through; `t` in the help screen |
| `finder/common.sh` (new) | Shared config (`LLMZIP_UV`, `LLMZIP_REPO`), `notify` (osascript or `LLMZIP_NOTIFY_CMD`), `log`, `run_llmzip`, `last_line` |
| `finder/llmzip-compress.sh` (new) | Compress action: skips folders and `.lz`; start and result notifications; `--unique` |
| `finder/llmzip-decompress.sh` (new) | Decompress action: `.lz` only; `--restored --unique`; IDENTICAL/DIFFERENT in the notification |
| `finder/llmzip-inspect.sh` (new) | Opens Terminal with `llmzip tui ABSOLUTE_FILE --tab heatmap`; `LLMZIP_DRY_RUN=1` prints the command |
| `finder/install.sh` (new) | Generates, lints and removes the 3 Quick Action bundles; `--dest DIR`, `--uninstall` |
| `finder/make_workflow.py` (new) | Writes `NAME.workflow/Contents/{Info.plist, document.wflow}` with stdlib `plistlib` |
| `tests/test_fileops.py` (new) | 19 naming, numbering and identity tests |
| `tests/test_cli.py` | +3: `--restored` IDENTICAL, DIFFERENT, not present, `--unique`, and the unchanged default |
| `tests/test_tui.py` | +5: switch on (IDENTICAL), DIFFERENT and not present, switch off, dotfiles and tree toggle, `--tab heatmap`; the existing round trip turns the switch off |
| `tests/test_finder.py` (new) | 7 script and installer tests |
| `README.md`, `CLAUDE.md` | Restored copies, `--unique`, `--tab` and `t`, "Finder actions" (install, actions, log, permissions, troubleshooting, manual Automator fallback); architecture notes |
| `reports/phase-7.md` | Commit-hash edit carried over from Phase 7 |
| `reports/phase-8.md` | This report |

## 4. Key decisions
- **Naming rules** (`fileops`):
  - `restored_output` inserts `.restored` before the **last** extension of the original's name:

    | input | restored copy |
    |---|---|
    | `report.md.lz` | `report.restored.md` |
    | `notes.lz` | `notes.restored` |
    | `a.tar.gz.lz` | `a.tar.restored.gz` |
    | `my notes.lz` | `my notes.restored` |
    | `.bashrc.lz` | `.bashrc.restored` |

    A non-`.lz` name raises `UsageError`.
  - `unique_path` has two rules, as you specified:
    - `where="base"` is used for `.lz` outputs. It numbers before all extensions, so `report.md.lz` → `report-2.md.lz` → `report-3.md.lz`, `a.tar.gz.lz` → `a-2.tar.gz.lz` and `notes.lz` → `notes-2.lz`.
    - `where="last"` is used for restored copies, so `report.restored.md` → `report.restored-2.md`.
  - So `report-2.md.lz` restores to `report-2.restored.md`, and the identity check then looks for `report-2.md`. Both are tested.
- **Identity check:**
  - The original is `restored.parent / (lz name minus ".lz")`, compared by size and SHA-256 (streamed).
  - It returns `None` when there's no original, or when the "restored" file *is* the original (restoring in place). In that case the CLI and TUI say "Original not present — CRC32 verified."
  - The line shows 12-hex SHA prefixes and `=` or `≠`.
- **CLI:** `--restored` and `--unique` are opt-in, so the default `decompress` behaviour and output are unchanged (tested). `-o` still wins over both naming rules.
  - **Small deviation:** you suggested the Finder scripts compute unique names and pass them with `-o`. I added `--unique` to the CLI instead, so the numbering exists only once, in Python, and the scripts stay simple.
- **TUI:**
  - The switch is on by default. An existing restored copy still gets the overwrite confirmation, as you specified; the TUI doesn't number files.
  - The identity check runs in the worker, since it hashes both files.
  - The result panel shows the line in **green** for IDENTICAL ✓ or **red** for DIFFERENT ✗, or dim for not present.
- **Finder scripts:**
  - They contain no logic beyond file-type checks and messages: all naming, coding and verification happens in the CLI, called as `uv run --project REPO --frozen --quiet llmzip …`.
  - Each script runs `cd` into the file's folder and passes the basename, so the CLI's summary line reads like `report.md -> report.md.lz mode=LLM 653 -> 154 B (ratio 4.24x, …)`, short enough for a notification.
  - Notification and Terminal text is passed to `osascript` through **argv** (`on run argv`), so apostrophes, quotes and unicode need no AppleScript escaping.
  - The inspect command quotes with zsh `${(qq)…}` and uses `${1:A}` to make the path absolute.
- **Installer:**
  - It bakes `LLMZIP_UV='/opt/homebrew/bin/uv'` and `LLMZIP_REPO='…'` into each Quick Action's command, because Automator's PATH is minimal.
  - I deliberately keep uv's **symlink** path. My first attempt resolved it to `/opt/homebrew/Cellar/uv/0.13.1/bin/uv`, which would break on the next `brew upgrade uv`.
  - Each bundle mirrors macOS's own Services bundles and Automator's Run Shell Script defaults, which I read from `/System/Library/Services/*.workflow` and the action's Info.plist:
    - zsh, input "as arguments";
    - `NSSendFileTypes = public.item`;
    - required context Finder;
    - `workflowTypeIdentifier = com.apple.Automator.servicesMenu`.
  - Both plists in every bundle are checked with `plutil -lint`. With the default destination, `pbs -flush` refreshes the Services menu.

## 5. Commands run and results
| Command | Result | Key output |
|---|---|---|
| Phase 7 baseline (`compress --mode llm` and auto, `bench/samples/*.txt`) from HEAD `82a37aa` before any edit | pass | 2,196 / 199 / 526 B |
| Naming spot-check | pass | every case in section 4 as specified |
| Manual Finder run (scratchpad, stub notifier) on `it's my notes.md` | pass | compress → `it's my notes.md.lz` (LLM 653 → 154 B); decompress → `it's my notes.restored.md`, **IDENTICAL ✓**, `cmp` OK; folder skipped; non-`.lz` file rejected; log written |
| `finder/install.sh --dest <scratchpad>` and `--uninstall` | pass | 3 bundles, 6 × `plutil -lint` OK, then removed; uv path fixed to `/opt/homebrew/bin/uv` |
| `uv run pytest tests/test_fileops.py tests/test_tui.py` | pass | 34 passed |
| `uv run pytest tests/test_finder.py` | pass | 7 passed in 25.8s |
| `uv run pytest` | pass | **368 passed** in 231.72s |
| Byte identity vs Phase 7 | pass | 6/6 identical |
| Headless TUI screenshot after a restore-as-copy decompress | pass | green IDENTICAL ✓ line; `.DS_Store` hidden; `t Tree` in the Footer. The cut-off switch label and wrapping status path it revealed were fixed, and the TUI tests re-run: 15 passed |

## 6. Smoke test results
**Byte identity vs Phase 7:**

| file | `--mode llm` | auto |
|---|---|---|
| blocktest.txt (2,196 B) | ✅ identical | ✅ identical |
| prose.txt (199 B) | ✅ identical | ✅ identical |
| prose_long.txt (526 B) | ✅ identical | ✅ identical |

**Recorded Finder notifications** (manual run, stub notifier):
```
llmzip | Compressing it's my notes.md… you'll get a notification when it's done.
llmzip ✓ compressed it's my notes.md | it's my notes.md -> it's my notes.md.lz  mode=LLM  653 -> 154 B  (ratio 4.24x, 1.895 bits/char)  1.3 s
llmzip | Skipped folder “finder1” (select files to compress).
llmzip | Decompressing it's my notes.md.lz… you'll get a notification when it's done.
llmzip ✓ restored it's my notes.md.lz | Original vs restored: IDENTICAL ✓  it's my notes.md (653 B, 0ac7ff98a8ec) = it's my notes.restored.md (653 B, 0ac7ff98a8ec)
                                        it's my notes.md.lz -> it's my notes.restored.md  mode=LLM  154 -> 653 B  (CRC32 ok)  1.3 s
llmzip ✗ stub.sh | Not a .lz file; only .lz files can be decompressed.
```

**Inspect dry run:**
```
cd '/Users/kianbahia/Stuff/Coding/llmzip' && '/opt/homebrew/bin/uv' run --frozen llmzip tui '/…/finder1/it'\''s my notes.md' --tab heatmap
```

**Generated Quick Action command:**
```
LLMZIP_UV='/opt/homebrew/bin/uv' LLMZIP_REPO='/Users/kianbahia/Stuff/Coding/llmzip' '/Users/kianbahia/Stuff/Coding/llmzip/finder/llmzip-compress.sh' "$@"
```

## 7. Test results
`368 passed in 231.72s`: Phase 7's 334 plus 34 new tests.
- **fileops (19):**
  - restored naming: 7 cases, including spaces, no extension, a dotfile and `report-2.md.lz`;
  - non-`.lz` names raise `UsageError` (3);
  - `last` numbering 2 and 3, plus `notes-2.restored`;
  - `base` numbering for `report.md.lz`, `a.tar.gz.lz`, `notes.lz` and `my notes.md.lz` (each to -2 and -3);
  - a numbered archive restores to `report-2.restored.md`; a bad `where` raises;
  - identity: identical, different, missing original, restoring in place, and the numbered-archive original `report-2.md`.
- **CLI (+3):**
  - `--restored` IDENTICAL ✓ on a real round trip;
  - `--unique` gives `report.restored-2.md` and `report-2.md.lz`, and the latter restores to `report-2.restored.md` with "Original not present";
  - an original modified before decompressing gives DIFFERENT ✗;
  - default decompress is unchanged, with no identity line.
- **TUI (+5):**
  - the switch defaults to on, and the result shows IDENTICAL ✓;
  - DIFFERENT when the original changed; then, with the original deleted, the confirm dialog, `y`, and "Original not present";
  - switch off restores in place;
  - `.DS_Store` and `.hidden` aren't in the tree, and `t` hides and shows it;
  - `tab="heatmap"` opens on the Heatmap with the analysis done (196 tokens) and the heatmap focused.
- **Finder (7):**
  - compress → decompress gives a byte-identical `report.restored.md`, with start, result and IDENTICAL notifications, and log entries;
  - `it's my notes.md` with an existing `.lz` and `.restored` → `it's my notes-2.md.lz`, then `…-2.restored.md`, then `…-2.restored-2.md`; nothing overwritten;
  - decompress with a non-`.lz` file then a real `.lz`: an error notification for the first, then IDENTICAL ✓;
  - compress skips a folder (with a space in its name) and a `.lz`;
  - an unreachable uv gives error notifications per file and the loop continues;
  - inspect dry run with a relative path containing an apostrophe and curly quotes: the command quotes the **absolute** path, and zsh reproduces it exactly;
  - `install.sh --dest tmp`: 3 bundles, `plutil -lint` OK, the expected NSServices and action parameters, the baked paths; `--uninstall` removes them all.

## 8. Deviations from the plan
- **`--unique` on the CLI** instead of computing names in the scripts and passing them with `-o`. See section 4 for why.
- **uv path:** the installer keeps uv's symlink path (`${UV:a}`, not `${UV:A}`), after I saw the Cellar path in my first test.
- **TUI label:** the switch label was shortened to "Restore as copy (.restored) + identity check", and the status line shows "Wrote NAME" instead of the full path. Both were found in a screenshot.
- **Inspect with several files:** only the first selected file is opened, and you get a notification saying so.

## 9. Problems and open questions
- **The Quick Actions themselves are untested in Finder.** I can't click in Finder, and I was told not to run `install.sh` against `~/Library`. The bundles lint and match the system's structure, but whether macOS 15.7 lists them on the first try is unverified. The README has the System Settings check, a Finder relaunch, and a click-by-click Automator fallback.
- **First-run prompts:** expect macOS prompts for file access (Documents, Desktop, Downloads) and notifications. Inspect may ask to control Terminal.
- **Notification length:** macOS truncates long notifications, so the decompress notification puts the identity line first.
- **Speed:** each Finder action starts a new `uv run` process, which costs about 3 s for the model load on the LLM path, plus the coding time. A large text file compresses for minutes; the start notification says to wait for the next one.
- **Numbered archives and the identity check:** a numbered archive such as `report-2.md.lz` is compared with `report-2.md`, which usually doesn't exist, so it says "original not present". That's correct per the rule, but worth knowing.

## 10. Notes for later phases
- Possible next steps:
  - a menu-bar or notification "Open in TUI" button after compress;
  - a `--restored` switch for `llmzip tui`'s decompress key binding from the CLI;
  - a signed or notarised app bundle, if the generated workflows prove fragile on future macOS versions.
- The project-level future ideas in CLAUDE.md are unchanged.

## 11. Git
- Commit: `<HASH>`
- Message: `Phase 8: restored copies, TUI polish, Finder actions`
- Staged explicitly, including the Phase 7 report hash edit. Not pushed.
- The hash was written into this file after the commit, so this line is an uncommitted edit.

## Manual test script (for your Mac)
1. **Install:** run `finder/install.sh`.
   - **Expect:** three `installed …/Library/Services/<Name>.workflow` lines and "Done."
   - If the actions don't show up in step 2, see README → Finder actions → Troubleshooting: enable them in System Settings, then `killall Finder`.
2. **Compress:** in Finder, open `testcompression/` and put a text file there (for example a copy of `bench/samples/prose.txt`). Right-click it → **Quick Actions** (or **Services**) → **Compress with llmzip**.
   - **Expect:** "Compressing prose.txt… you'll get a notification when it's done", then "llmzip ✓ compressed prose.txt — prose.txt -> prose.txt.lz mode=LLM 1301 -> 199 B (ratio 6.54x …)".
   - `prose.txt.lz` appears. Allow any file-access or notification prompts.
3. **Compress again:** right-click the same file → Compress with llmzip.
   - **Expect:** `prose-2.txt.lz`; the first `.lz` is untouched.
4. **Decompress:** right-click `prose.txt.lz` → **Decompress with llmzip**.
   - **Expect:** a notification titled "llmzip ✓ restored prose.txt.lz" with **"Original vs restored: IDENTICAL ✓ prose.txt (1,301 B, …) = prose.restored.txt (1,301 B, …)"**.
   - `prose.restored.txt` appears. Running it again gives `prose.restored-2.txt`.
   - **Optional:** edit `prose.txt`, then decompress again: the title says "DIFFERENT from the original" and the line shows ✗.
5. **Inspect:** right-click `prose.txt` → **Inspect with llmzip**.
   - **Expect:** Terminal opens and runs `llmzip tui …/prose.txt --tab heatmap`; the heatmap analysis starts by itself.
   - Press `t` to hide the tree; `q` quits.
6. **TUI switch:** run `uv run llmzip tui testcompression`. Note that no `.DS_Store` is listed. Select `prose.txt.lz` and press **ctrl+d** with **Restore as copy** on.
   - **Expect:** the result panel shows the green **IDENTICAL ✓** line (confirm the overwrite if `prose.restored.txt` exists).
   - Turn the switch off and press **ctrl+d**: you're asked to overwrite `prose.txt` (restore in place).
7. **Log:** `tail ~/Library/Logs/llmzip/finder.log` shows every command, its exit code and its output.
8. **Uninstall (optional):** `finder/install.sh --uninstall` removes the three actions.
