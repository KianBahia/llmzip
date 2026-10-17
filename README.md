# llmzip

Lossless file compression with a local GPT-2 model and arithmetic coding. Predictable text costs a fraction of a bit per token, so English prose comes out 3–4× smaller than gzip, xz or zstd. Runs on CPU only and is fully deterministic, offline after the first run.

## Quickstart

```
# 1. Install (needs uv: https://docs.astral.sh/uv/)
git clone <this repo> && cd llmzip
uv sync

# 2. First run downloads GPT-2 (~500 MB) into the Hugging Face cache; later runs are offline
uv run llmzip info --fingerprint

# 3. Use it
uv run llmzip compress notes.txt                 # -> notes.txt.lz (auto: smallest of LLM / zstd / stored)
uv run llmzip decompress notes.txt.lz -o notes.restored.txt
uv run llmzip inspect notes.txt                  # colour heatmap of how surprising each token was
uv run llmzip bench bench/samples/*              # compare against gzip -9 / xz -9 / zstd -19
uv run llmzip tui bench/samples                  # interactive terminal app (see "Terminal app")
finder/install.sh                                # optional: right-click actions in Finder (see "Finder actions")
```

**compress** options:
- `--mode auto|llm|zstd|store` (default `auto`: smallest wins);
- `--max-llm-tokens N` (default 25000; auto mode uses zstd above this, `0` = unlimited);
- `--no-skip` to always try the full LLM in auto mode;
- `--verify`, `-o OUT`, `--force`, `-q`.

**inspect** options: `--top N`, `--max-tokens N` (default 5000), `--block-size N`, `--json OUT`, `--no-color`.

**decompress** options: `-o OUT`, `--force`, `-q`, plus:
- `--restored`: write a copy named `NAME.restored.EXT` (for example `report.md.lz` → `report.restored.md`). If the original is still there, it prints `Original vs restored: IDENTICAL ✓` (or `DIFFERENT ✗`) with sizes and SHA-256s.
- `--unique`: never fail on an existing output; use `report.restored-2.md`, `-3`, … instead.

`compress --unique` numbers the archive before its extensions (`report-2.md.lz`), so a later `--restored` still gives `report-2.restored.md`.

Exit codes are 0 ok, 1 error, 2 usage error. Set `LLMZIP_DEBUG=1` for tracebacks.

**Limitations, in short:**
- Slow (about 100–150 tokens/s, roughly 4 KB of text per 10 s).
- Decompression needs the identical model, library versions and machine type.
- Only valid UTF-8 text benefits; everything else goes to zstd or is stored. See [Limitations](#limitations).

## How it works

GPT-2 reads the text one token at a time and predicts a probability for every possible next token. Those probabilities are turned into integer frequency tables, and an integer arithmetic coder spends about `-log2 p` bits on the token that actually comes next. Well-predicted tokens cost a fraction of a bit.

Decompression runs the same model with the same settings, rebuilds the same tables, and decodes the tokens back in order. Each piece fits into the next:
- Text is coded in independent blocks of up to 1023 tokens (GPT-2's context).
- Non-text, or text where zstd is predicted to win, goes to zstd or is stored as-is.
- Every file carries a CRC32 and a fingerprint of the model and software that produced it.

## Results

`uv run llmzip bench bench/samples/*` on an Apple M4 Pro (CPU, 1 thread). All sizes are in bytes and include llmzip's 40+ byte container header. The classic tools read from stdin, so no file name is stored. Every llmzip file was round-tripped and verified.

| file | original | **llmzip** | bits/byte | gzip -9 | xz -9 | zstd -19 | vs best classic |
|---|---|---|---|---|---|---|---|
| prose.txt (English prose) | 1,301 | **199** | 1.22 | 697 | 820 | 702 | 3.5× smaller |
| prose_long.txt (English prose) | 4,150 | **526** | 1.01 | 1,909 | 2,004 | 1,872 | 3.6× smaller |
| sample.json | 775 | **150** | 1.55 | 365 | 428 | 368 | 2.4× smaller |
| sample.md | 653 | **154** | 1.89 | 424 | 508 | 427 | 2.8× smaller |
| sample.py | 761 | **207** | 2.18 | 444 | 540 | 452 | 2.1× smaller |
| blocktest.txt (technical markdown) | 9,515 | **2,196** | 1.85 | 4,399 | 4,352 | 4,334 | 2.0× smaller |
| reports_concat.md (technical markdown) | 32,883 | **7,419** | 1.80 | 12,763 | 12,104 | 12,351 | 1.6× smaller |

The full results are in `bench/results/`.

## Inspect

`llmzip inspect FILE` shows what compression "sees". It uses the same tokens, the same 1023-token blocks and the same quantized probabilities, so the bits it reports are the bits actually paid. Example on `bench/samples/prose.txt`, in plain text (`--no-color --top 5`):

```
tokens 284  characters 1301  blocks 1  bits 1255.8 (ideal 1254.9)  bits/char 0.965
predicted LLM payload 157 B  actual payload 157 B  predictability 18.7% of tokens < 1 bit

                         Top 5 most surprising tokens
  #  token        bits  prob     context                                              model's top 3
 89  ' no'        16.8  8.6e-06  ttle further every spring[ no] matter how oft        '.' 0.67, ',' 0.07, ' when' 0.06
 49  ' strings'   16.3  1.2e-05  and beans that climbed up[ strings] she tied to th   ' the' 0.35, ' to' 0.11, ' into' 0.05
173  ' enormous'  15.0  3e-05    d just laugh, hold up the[ enormous] vegetable, and  ' basket' 0.29, ' carrot' 0.02, ' little' 0.02
254  ' reason'    13.9  6.6e-05  eally missed was having a[ reason] to go outside     ' garden' 0.35, ' good' 0.07, ' little' 0.02
 77  ' stubborn'  13.6  8.3e-05  l, thyme, rosemary, and a[ stubborn] clump of mint   ' few' 0.17, ' little' 0.15, ' handful' 0.03
```

It also lists the longest "autopilot stretches", runs of consecutive tokens that each cost under 0.5 bits.

### How to read the heatmap

In a terminal, the file is printed exactly as it is, with every token's background coloured by how many bits it cost:

- **Dark blue:** under 0.5 bits.
- **Light blue:** 0.5–1.5 bits.
- **Pale yellow:** 1.5–3 bits.
- **Orange:** 3–6 bits.
- **Red-orange:** 6–12 bits.
- **Dark red:** 12 or more bits.

Blue stretches are text the model saw coming. Boilerplate, closing brackets and the second half of a word are nearly free. Red marks real information: names, numbers, unusual word choices. Whitespace tokens are coloured too, so expensive indentation or newlines show up as coloured gaps.

"Predicted payload" is the sum of these bits per block, rounded up. It matches the real compressed payload to within a few bytes.

## Terminal app

`llmzip tui [PATH]` opens an interactive app. PATH can be a folder, which opens the file browser there, or a file, which opens it directly; the default is the current directory. It does the same work as the CLI (same compression, same files and the same heatmap) with a mouse- and keyboard-friendly interface. Slow jobs run in the background and can be cancelled.

```
uv run llmzip tui bench/samples              # browse the samples
uv run llmzip tui notes.txt --max-tokens 2000
uv run llmzip tui notes.txt --tab heatmap      # open straight on a tab (heatmap/stats analyze automatically)
```

Dotfiles such as `.DS_Store` are hidden in the file tree. Press `t` to hide the tree and give the heatmap the full width.

| key | action |
|---|---|
| `1` `2` `3` `4` | Overview / Heatmap / Stats / Compare tab |
| `tab` / `shift+tab` | move focus between the file tree, buttons, heatmap and tables |
| `ctrl+o` | compress the selected file (mode from the Overview selector) |
| `ctrl+d` | decompress the selected `.lz` file |
| `v` | verify: round trip in memory, nothing written |
| `a` | analyze: build the heatmap and stats |
| `b` | run the benchmark comparison |
| `c` | cancel the running job (nothing is written) |
| `t` | show or hide the file tree |
| `← → ↑ ↓`, mouse click | heatmap: move the token cursor |
| `enter` (in the Stats table) | jump to that token in the heatmap |
| `?` | help |
| `q` | quit |

**Walkthrough:**
1. Pick a file in the tree. The **Overview** shows its size, type and token count; for a `.lz` file it shows the header and whether the fingerprint matches.
2. Choose a mode (auto, llm, zstd or store) and press **Compress**. A progress bar shows tokens done, tok/s and ETA, and the result panel shows the chosen mode, sizes, ratio and bits/char, with any "LLM skipped" reason.
   - If the output already exists, you're asked before anything is overwritten.
   - Select the `.lz` and press **Decompress** to get the file back.
   - With **Restore as copy (.restored)** on (the default), decompress writes `NAME.restored.EXT` next to the original, and the result panel shows **Original vs restored: IDENTICAL ✓** (or **DIFFERENT ✗**) with sizes and short SHA-256s. Turn the switch off to restore in place, as `NAME.EXT`.
3. Open **Heatmap** (`2`). The analysis runs on demand and colours every token by its cost, using the same scale as `llmzip inspect`. Move the cursor with the arrow keys or click a token; the side panel shows its bits, probability, block and the model's top-5 guesses.
4. **Stats** (`3`) has the summary, a histogram of bit buckets, the most surprising tokens (press enter to jump to one in the heatmap) and the autopilot stretches.
5. **Compare** (`4`) runs llmzip against gzip, xz and zstd for this file and highlights the smallest.

## Finder actions (macOS)

Three right-click Quick Actions: **Compress with llmzip**, **Decompress with llmzip** and **Inspect with llmzip**.

```
finder/install.sh               # install into ~/Library/Services
finder/install.sh --uninstall   # remove them
```

What each action does:
- **Compress with llmzip:** compresses each selected file (auto mode) to `NAME.EXT.lz`. It never overwrites; it uses `NAME-2.EXT.lz` and so on instead. Folders and `.lz` files are skipped.
- **Decompress with llmzip:** restores each selected `.lz` as a copy, `NAME.restored.EXT` (numbered if it exists). If the original is still next to it, the notification says **IDENTICAL ✓** or **DIFFERENT ✗**.
- **Inspect with llmzip:** opens Terminal with `llmzip tui FILE --tab heatmap`.

Files are processed one after another. You get a notification when each file starts and finishes (mode, sizes and ratio), or a clear error, and a failure on one file doesn't stop the rest. Every run is logged to `~/Library/Logs/llmzip/finder.log`.

**First run.** macOS may ask you to allow Finder or Automator to access files in Documents, Desktop or Downloads, and to allow notifications. If no notifications appear, check System Settings → Notifications. Inspect may also ask to let the action control Terminal.

**Troubleshooting:**
- **The actions don't appear after install:** enable them in System Settings. Depending on the macOS version this is under General → Login Items & Extensions → Finder (Quick Actions), or under Keyboard → Keyboard Shortcuts → Services → Files and Folders. Then relaunch Finder (`killall Finder`).
- **You moved the project:** the actions contain the absolute paths of this repo and of `uv`, so rerun `finder/install.sh` after moving it.
- **Manual setup, if the generated actions don't work on your macOS version:**
  1. Open Automator, choose File → New → **Quick Action**.
  2. Set "Workflow receives current" to **files or folders** in **Finder**.
  3. Add **Run Shell Script**, with Shell **/bin/zsh** and Pass input **as arguments**.
  4. Replace the script with the line below, adjusting both paths (and `compress` → `decompress` / `inspect` for the other actions):

     ```
     LLMZIP_UV=/opt/homebrew/bin/uv LLMZIP_REPO=/path/to/llmzip /path/to/llmzip/finder/llmzip-compress.sh "$@"
     ```
  5. Save it as "Compress with llmzip".

## Limitations

- **Slow:** about 100–150 tokens/s in each direction on one CPU core (about 4 KB of text per 10 s); 1 MB of text takes roughly half an hour.
  - Auto mode only uses the LLM up to `--max-llm-tokens` (25000 by default) and uses zstd above that.
  - In auto mode, text that zstd will clearly win on (for example, highly repetitive logs) is detected from the first block, and the LLM is skipped.
  - Text that GPT-2 predicts well still gets the full, slow LLM run.
- **Decompression needs the identical setup.** That means the same GPT-2 weights, the same torch, transformers and numpy versions, and the same settings; the fingerprint check enforces this. In practice it also means the same kind of machine: bit-exact results across different CPUs or operating systems are not guaranteed, and haven't been tested.
- **Only valid UTF-8 text** benefits. Everything else falls back to zstd or is stored.
- Each 1023-token block starts with no context, which costs a little compression on long files.

## Development

```
uv run pytest
uv run python scripts/demo_cli.py             # CLI end to end on samples + binaries
uv run python scripts/demo_llm.py bench/samples/*
uv run llmzip bench bench/samples/* [--block-sizes 1023,512] [--no-decompress]
uv run python scripts/profile_step.py         # per-token time breakdown
```
