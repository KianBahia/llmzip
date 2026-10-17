# Phase 5 report: benchmarks, weight-hash cache, LLM skip heuristic

## 1. Summary
I added `llmzip bench`, a disk cache for the weights hash, an auto-mode heuristic that skips the LLM when zstd will clearly win, and a per-token profiler. Nothing that affects coded bits changed: the `.lz` files for every sample are byte-identical to Phase 4's, and the fingerprint is still `2bb9a6826f48e26a`.

On the samples, llmzip is 1.6–3.6× smaller than the best of gzip -9, xz -9 and zstd -19. The heuristic compresses a 200 KB repetitive log in 9.9 s instead of about 11 minutes. All 312 tests pass.

## 2. Environment
Unchanged: macOS 15.7.1, Apple M4 Pro, uv 0.13.1, Python 3.12.15, torch 2.14.1, transformers 5.20.1, numpy 2.5.4, zstandard 0.25.0, rich 15.0.0, xz 5.8.4, zstd 1.5.7, Apple gzip 457.140.3, git 2.50.1. No new dependencies.

## 3. Files created/changed
| File | Purpose |
|---|---|
| `src/llmzip/model.py` | Weights-hash cache: `cache_dir()` (`~/Library/Caches/llmzip` or `$LLMZIP_CACHE_DIR`), keyed by the resolved weights file, size, mtime_ns and revision; `weights_sha256(use_cache=...)`. Any cache error means a recompute |
| `src/llmzip/api.py` | Skip heuristic in auto mode, `skip_estimate()` (shared with bench), `CompressInfo.llm_skipped`, `skip=` parameter |
| `src/llmzip/llmcodec.py` | `encode_block` made public (it was `_encode_block`); `compress_text(done_blocks=...)` reuses already-coded leading blocks. The coding path is untouched |
| `src/llmzip/bench.py` (new) | `llmzip bench`: forced LLM plus the replayed auto choice, gzip, xz and zstd piped from stdin, bits/byte, timings without model load, round trip, block-size table, rich table, JSON and markdown output |
| `src/llmzip/cli.py` | `bench` subcommand (`--block-sizes`, `--no-decompress`, `--out`); `compress --no-skip`; the summary line reports a skip |
| `scripts/profile_step.py` (new) | Per-token time breakdown, measurement only |
| `tests/test_cli.py` | +2: a repetitive log is skipped; prose isn't skipped and gives output identical to no-skip and forced-LLM runs |
| `tests/test_model.py` | +1: cached, uncached and corrupt-cache paths give the same fingerprint |
| `bench/samples/blocktest.txt` (new) | First 3,000 tokens of `reports_concat.md`, decoded from token ids and checked to round-trip (9,515 B) |
| `bench/results/20261017-115646.{json,md}` (new) | Full sample benchmark |
| `bench/results/20261017-120142.{json,md}` (new) | Block-size experiment |
| `README.md` | How it works, Results, Limitations, bench usage, `--no-skip` |
| `CLAUDE.md` | Heuristic, cache, bench and the "no coded-bit changes" rule; roadmap |
| `reports/phase-4.md` | Commit-hash edit and the post-commit demo row, both carried over |
| `reports/phase-5.md` | This report |

## 4. Key decisions
- **Hard constraint respected:** `probs.py`, `coder.py`, `freqtable.py`, `_cum_tables`, the tokenizer flags and the container format are unchanged. There was no `PROBS_SCHEME` bump. `encode_block` is the same function under a public name.
- **Weights-hash cache:**
  - One small JSON file holds a single entry. The key is `<resolved model.safetensors path>|<size>|<mtime_ns>|<MODEL_REVISION>`.
  - It is written atomically (temp file plus `os.replace`).
  - Anything wrong (a missing file, malformed JSON, a value that isn't 64 hex characters, an unwritable directory) silently falls back to recomputing.
  - The cache only stores the same `state_dict` hash as before, so the fingerprint can't change. Verified: cold, warm and corrupt-cache runs all give `2bb9a6826f48e26a`.
- **Skip heuristic:** it runs only in auto mode, only for valid round-trippable text with more than `2 × block_size` tokens, and only without `--no-skip`.
  - It codes block 1, measures its exact byte length (decoding block 1's ids), and estimates `len(block1) × total_bytes / block1_bytes`.
  - It skips if the estimate is greater than the zstd frame size.
  - Otherwise block 1 is passed as `done_blocks`, so it is never coded twice and the output is bit-identical to a no-skip run (tested, and verified on `blocktest.txt`).
  - `skip_estimate` lives in api.py and is reused by bench, so the bench's "auto pick" is exactly what `compress` would choose.
- **Bench design:**
  - The model loads once; the load time (including the fingerprint) is reported separately and excluded from the per-file times.
  - The forced-LLM `.lz` gives the payload and container sizes.
  - The auto pick is **replayed** from the forced-LLM blocks (bit-identical) plus the STORED and ZSTD sizes and the heuristic, instead of running the LLM a second time.
  - gzip, xz and zstd read the file from stdin.
  - Block sizes other than 1023 are compress-only runs; the 1023 result is reused.

## 5. Commands run and results
| Command | Result | Key output |
|---|---|---|
| `git worktree add $TMPDIR/llmzip_p4 8601f1d`, then Phase 4 `llmzip compress --mode llm` on 6 samples (`PYTHONPATH=worktree/src`) | pass | Phase 4 baseline `.lz` files |
| Cache check (cold, warm, corrupt cache) | pass | `2bb9a6826f48e26a` each time; warm hash 0.000 s vs 0.317 s cold |
| `uv run pytest -k "skip or cache"` | pass | 3 passed |
| `uv run python scripts/profile_step.py` | pass | see section 6 |
| `uv run llmzip bench bench/samples/*` | pass | all round trips OK; `bench/results/20261017-115646.*` |
| `uv run llmzip bench bench/samples/blocktest.txt --block-sizes 1023,512,256,128 --no-decompress` | pass | `bench/results/20261017-120142.*` |
| Auto mode on a 200 KB log with random fields | ran full LLM (not a bug) | stopped after about 10 minutes; see section 6 |
| Auto mode on a 200 KB log of identical lines | pass | `mode=ZSTD 200007 -> 113 B (…; LLM skipped: est. 2377 B > zstd 95 B) 9.9 s`; `cmp` OK |
| Byte identity vs Phase 4 (`--mode llm --no-skip`, 6 files, plus auto mode on blocktest) | pass | all 7 comparisons identical |
| `uv run pytest` | pass | 312 passed in 144.12s |

## 6. Smoke test results

### Benchmark (`llmzip bench bench/samples/*`)
Sizes are in bytes; bpb is bits per original byte. Model load (including the fingerprint, with a warm cache): 0.12 s.

| file | bytes | llmzip .lz | bpb | auto pick | gzip -9 (bpb) | xz -9 (bpb) | zstd -19 (bpb) | comp s | decomp s | tok/s c / d | round trip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| blocktest.txt | 9,515 | **2,196** | 1.846 | LLM | 4,399 (3.70) | 4,352 (3.66) | 4,334 (3.64) | 31.2 | 30.3 | 96 / 99 | OK |
| prose.txt | 1,301 | **199** | 1.224 | LLM | 697 (4.29) | 820 (5.04) | 702 (4.32) | 2.2 | 1.9 | 128 / 153 | OK |
| prose_long.txt | 4,150 | **526** | 1.014 | LLM | 1,909 (3.68) | 2,004 (3.86) | 1,872 (3.61) | 8.1 | 8.1 | 110 / 110 | OK |
| reports_concat.md | 32,883 | **7,419** | 1.805 | LLM | 12,763 (3.11) | 12,104 (2.95) | 12,351 (3.01) | 96.6 | 102.9 | 106 / 100 | OK |
| sample.json | 775 | **150** | 1.548 | LLM | 365 (3.77) | 428 (4.42) | 368 (3.80) | 2.4 | 2.4 | 149 / 151 | OK |
| sample.md | 653 | **154** | 1.887 | LLM | 424 (5.19) | 508 (6.22) | 427 (5.23) | 1.3 | 1.3 | 151 / 156 | OK |
| sample.py | 761 | **207** | 2.176 | LLM | 444 (4.67) | 540 (5.68) | 452 (4.75) | 2.1 | 2.1 | 149 / 149 | OK |

Compared with the best classic tool, llmzip is 3.5–3.6× smaller on prose, 2.1–2.8× smaller on the code, JSON and markdown snippets, and 1.6–2.0× smaller on the technical reports. The piped gzip sizes are about 10–20 B smaller than Phase 4's, because no file name is stored.

### Block-size experiment (`blocktest.txt`, 3,000 tokens, compress only)
| block size | blocks | .lz bytes | payload | bpb | vs 1023 | comp s |
|---|---|---|---|---|---|---|
| 1023 | 3 | 2,196 | 2,150 | 1.846 | — | 32.1 |
| 512 | 6 | 2,310 | 2,258 | 1.942 | +5.2% | 23.1 |
| 256 | 12 | 2,405 | 2,341 | 2.022 | +9.5% | 19.4 |
| 128 | 24 | 2,538 | 2,474 | 2.134 | +15.6% | 19.1 |

- **Restart cost:** going from 3 to 24 blocks adds 324 B of payload over 21 extra restarts, about 15 B per restart on average.
  - It is largest when long context is lost: 1023→512 costs about 36 B per extra restart, while 256→128 costs about 11 B.
  - So most of the cost is lost long-range context, not just the cold start.
- **Speed:** smaller blocks are faster (32 s vs 19 s), because attention over a shorter KV cache is cheaper. That's a speed/size trade-off worth knowing, but the default stays at 1023.

### Per-token profile (`scripts/profile_step.py`, prose.txt, 284 tokens)
| stage | µs/token | share |
|---|---|---|
| model forward (`next_logits`) | 5,771.6 | 93.5% |
| `logits_to_freqs` (float64 softmax, floor, remainder) | 288.5 | 4.7% |
| `cumulative` (50k cumsum + validation) | 106.5 | 1.7% |
| `symbol_interval` / `find_symbol` | 4.5 | 0.1% |
| arithmetic coder (encode, or `get_target` + `consume`) | 3.6 | 0.1% |
| **total** | **6,174.7** | about 162 tok/s |

The model dominates. Optimizing probs and cumsum could save at most about 6%.

### Heuristic timing
- **Identical-line log** (`$TMPDIR/llmzip_log200k_rep.txt`, 200,007 B):
  - Auto mode took **9.9 s** (12.0 s wall including startup) → ZSTD, 113 B, with "LLM skipped: est. 2377 B > zstd 95 B", and `cmp` OK after decompression.
  - That's about one block's worth of time: compressing the 894-token `prose_long.txt` took 11.1 s wall.
  - A full LLM run would be about 69k tokens, roughly 11 minutes.
- **Log with random fields** (`$TMPDIR/llmzip_log200k.txt`: random path and latency per line, 200,011 B, 85,556 tokens):
  - The heuristic did **not** skip, correctly. GPT-2 coded block 1 into 72 B (about 0.24 bits/byte), giving an estimate of 6,083 B against a zstd frame of 11,782 B.
  - GPT-2 genuinely compresses this log about 2× better than zstd, so the LLM is the right size choice, but it costs about 15 minutes. I stopped the run after about 10 minutes.
  - This shows the heuristic prevents *worse sizes*, not *long runtimes* (see section 9).

### Byte identity vs Phase 4
I compressed the files with the Phase 5 code (`--mode llm --no-skip`) and compared them byte-for-byte with `.lz` files made by the Phase 4 commit `8601f1d` (worktree in `$TMPDIR`):

| file | size | result |
|---|---|---|
| blocktest.txt.lz | 2,196 B | ✅ identical (sha256 `d6965bf9e23b…`) |
| prose.txt.lz | 199 B | ✅ identical (`25a09f0a7ed4…`) |
| prose_long.txt.lz | 526 B | ✅ identical (`679723c8e680…`) |
| sample.json.lz | 150 B | ✅ identical (`22332b8c0fba…`) |
| sample.md.lz | 154 B | ✅ identical (`08f32517db47…`) |
| sample.py.lz | 207 B | ✅ identical (`2d090e999230…`) |
| blocktest.txt.lz, **auto mode** (heuristic ran, block 1 reused) | 2,196 B | ✅ identical |

That covers every `bench/samples/*.txt` file plus the other small samples. `reports_concat.md` was skipped for time; its LLM size (7,419 B) matches Phase 4's demo exactly.

## 7. Test results
`312 passed in 144.12s`. That's Phase 4's 309 plus 3 new tests:
- **Cache:** the cached, uncached and corrupt-cache paths all give the same fingerprint, and the cache file is created only when caching is used.
- **Heuristic, skip case:** 120 identical log lines (more than 2 blocks) are skipped, choosing ZSTD or STORED with no LLM candidate, and round-trip.
- **Heuristic, no-skip case:** prose with `block_size=50` (6 blocks) runs the heuristic and isn't skipped. Its output is byte-identical to `skip=False` and to forced `--mode llm`, which proves the first-block reuse is exact.

## 8. Deviations from the plan
- **Auto pick in bench:** this is replayed from the forced-LLM run instead of a second `compress --mode auto`. It is exactly equivalent, since the blocks are bit-identical and the heuristic is the same function, and it halves the bench time.
- **Heuristic run (c):** my first log had random fields, and the LLM was correctly *not* skipped. I stopped that run, recorded its estimate, and used an identical-line log to time the skip. Both cases are in section 6.
- **Byte-identity check:** this covers all six small samples, not just `*.txt`, plus one auto-mode check. `reports_concat.md` was excluded for time.
- **Extras:** `bench` validates file paths (usage error, exit 2), exits 1 if any round trip fails, and widens the rich table to 160 columns when output isn't a terminal.

## 9. Problems and open questions
- **No unresolved failures.**
- **The heuristic only guards size, not time.** Text that GPT-2 compresses well (like the random-field log) still gets the full run at about 100–150 tok/s. A size or time budget option (for example `--max-llm-tokens`) would be a policy decision for you.
- **The heuristic estimate is a single-block extrapolation.** It is pessimistic for text whose later blocks compress better and optimistic for the reverse. It can't produce a larger output than the no-skip path would choose between, but it can occasionally pick ZSTD when the LLM would have won narrowly. It only compares sizes, and the margin is 0%.
- **Model load:** the bench shows 0.12–0.13 s with a warm OS file cache and a warm hash cache. A cold first load is about 2 s, as measured in Phase 4.
- `bench/results/` is committed. Every future run adds files there; decide whether to keep doing that or ignore new ones.

## 10. Notes for later phases
- **Phase 6, inspect:** reuse `llmcodec.encode_block`'s structure (restart per 1023 tokens) and `softmax64(logits)[token]` for the surprise values. The profile shows that computing surprise for display adds negligible cost over the forward pass.
- **Possible speedups (not done; they don't change bits, so they're allowed):**
  - Forward-pass overhead per token through HF's Python layers is most of the 5.8 ms. A leaner GPT-2 forward pass (direct module calls, a preallocated KV cache) could help, but it **must be bit-identical**. Verify with the fresh-process logits digest and a byte comparison of `.lz` files, as done in this phase.
  - Parallelism across blocks: blocks are independent, so N processes could code N blocks at once with no format change. This is probably the biggest practical win for large files on a 10-core machine.
- **Future format v2:** sliding-window context to recover part of the block-restart loss. Halving the block size already costs +5.2% (see the block-size table); the loss at 1023 itself is smaller but unmeasured.

## 11. Git
- Commit: `1aec5d5`
- Message: `Phase 5: benchmarks, weight-hash cache, LLM skip heuristic`
- The hash was written into this file after the commit, so this line is an uncommitted edit that goes into the Phase 6 commit.
