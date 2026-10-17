# Phase 1 report: project setup and model smoke test

## 1. Summary
I set up the uv project (Python 3.12) with the planned structure, a working `info` command and stubs for later phases. `NextTokenModel` runs GPT-2 deterministically on CPU, one token at a time with a KV cache. Incremental logits are bit-identical across repeated runs, across thread counts 1, 4 and 10, and across fresh processes. The ideal compressed size of a 210-word paragraph is 144 bytes (0.989 bits/char), against 637 bytes for `gzip -9`.

## 2. Environment
| Item | Version |
|---|---|
| macOS | 15.7.1 (build 24G231) |
| Chip | Apple M4 Pro (10 performance + 4 efficiency cores) |
| uv | 0.13.1 (Homebrew 2026-10-14) |
| Python | 3.12.15 (uv-managed) |
| xz | 5.8.4 (liblzma 5.8.4) |
| zstd | 1.5.7 |
| gzip | Apple gzip 457.140.3 |
| git | 2.50.1 (Apple Git-155) |
| torch | 2.14.1 |
| transformers | 5.20.1 |
| huggingface_hub | 2.2.0 (transformers dependency) |
| zstandard | 0.25.0 |
| rich | 15.0.0 |
| pytest | 9.1.1 |

## 3. Files created/changed
| File | Purpose |
|---|---|
| `pyproject.toml` | Project metadata, dependencies, `llmzip` console script, pytest config with the `slow` marker |
| `uv.lock` | Pins the exact dependency versions |
| `.python-version` | Pins Python 3.12 for uv |
| `.gitignore` | Ignores `.venv`, `__pycache__`, `.pytest_cache`, `bench/corpus/`, `*.lz` and `.DS_Store` |
| `CLAUDE.md` | Overview, HARD RULES (verbatim), conventions and phase roadmap |
| `README.md` | Short description and run commands |
| `src/llmzip/__init__.py` | Package docstring and `__version__` |
| `src/llmzip/config.py` | `MODEL_NAME`, `MODEL_REVISION`, `ATTN_IMPLEMENTATION`, `BOS_TOKEN_ID`, `NUM_THREADS=1`, `FREQ_TOTAL_BITS=16` |
| `src/llmzip/model.py` | `NextTokenModel` (deterministic incremental GPT-2, fingerprint), `logits_digest()`, and a `python -m llmzip.model N` digest entry point for fresh-process checks |
| `src/llmzip/cli.py` | argparse CLI: working `info [--fingerprint]`; `compress`, `decompress`, `inspect` and `bench` print "not implemented yet" and exit 2 |
| `src/llmzip/probs.py`, `coder.py`, `container.py`, `fallback.py`, `inspect_cmd.py` | Docstring-only stubs naming the phase that fills each one |
| `scripts/smoke_model.py` | Determinism (in-process, across thread counts, across processes), speed, batched-vs-incremental comparison, ideal size vs gzip/xz/zstd |
| `tests/test_model.py` | 7 tests (see section 7) |
| `bench/.gitkeep` | Placeholder for the benchmark directory |
| `reports/phase-1.md` | This report |

## 4. Key decisions
- **Start of sequence:** `start()` resets the model, feeds `<|endoftext|>` (id 50256) as BOS and returns the logits for the first real token. That's what GPT-2 was trained to see between documents, so the first token gets a real distribution instead of a hard-coded uniform one. It costs one extra forward pass per file.
- **`logits_sequence(ids)`:** one shared function that returns exactly one logits tensor per token in `ids`. The logits for `ids[i]` come from the model after it has seen BOS plus `ids[:i]`. The last token is never fed, because nothing follows it (the original length will be stored in the header). This is the single function phases 2–3 must use on both the encode and decode sides; the decoder will call `start()`/`next_logits()` step by step in the same order.
- **KV cache:** the `past_key_values` (a transformers `DynamicCache`) returned by each forward pass is passed into the next one. Each call feeds a `[[token_id]]` tensor. The returned logits are `.clone()`d so later calls can't change them.
- **Context limit:** GPT-2 has 1024 positions, and BOS takes one, so 1023 text tokens fit before `next_logits` raises `ValueError`. Handling long inputs is a later-phase decision.
- **Attention:** `attn_implementation="eager"` is forced, so the math doesn't depend on which SDPA backend a given torch version picks. It's also part of the fingerprint.
- **Revision pinning:** `MODEL_REVISION = 607a30d783dfa663caf39e06633721c8d4cfcd7e`, fetched with `huggingface_hub.model_info("gpt2").sha`; the Hub reports `last_modified` as 2024-02-19. It isn't from memory. The tokenizer and the weights both load at this revision.
- **Fingerprint:** the first 16 hex characters of SHA-256 over `model name | revision | SHA-256 of all weights (state_dict sorted by name; names + raw bytes) | torch version | transformers version | threads=N | attn=eager`. The weights hash is computed once and cached. The current value at 1 thread is `bf3b4c5f19722425`. The thread count stays in the fingerprint even though logits matched across thread counts here, because that may not hold for other inputs or builds.
- **Determinism settings** are applied in the constructor: `set_num_threads`, `use_deterministic_algorithms(True)` and `set_grad_enabled(False)`, plus `@torch.no_grad()` on `next_logits`. The model is float32 on CPU in `.eval()` mode.
- **`NUM_THREADS = 1`:** the fastest setting measured, and fully deterministic (see section 6).
- **Python:** uv installed its own managed CPython 3.12.15. The only other 3.12 on the machine was miniconda's, and I kept conda out of the project.

## 5. Commands run and results
| Command | Result | Key output |
|---|---|---|
| `sw_vers`, `sysctl …`, `uv/xz/zstd/gzip/git --version` | pass | see section 2 |
| `git init -b main` | pass | |
| `uv python install 3.12` | pass | CPython 3.12.15 installed |
| `uv init --package --name llmzip --python 3.12` | pass | src-layout project |
| `uv add torch transformers zstandard rich` / `uv add --dev pytest` | pass | versions in section 2 |
| `uv run python -c "…model_info('gpt2').sha"` | pass | `607a30d7…cfcd7e` |
| `uv run python scripts/smoke_model.py` (with NUM_THREADS=4, then 1) | pass (exit 0) | see section 6 |
| `uv run pytest` | pass | 7 passed in 7.53s |
| `uv run llmzip info` | pass (exit 0) | prints config, torch 2.14.1, threads 1, gpt2 @ revision |
| `uv run llmzip info --fingerprint` | pass | `fingerprint: bf3b4c5f19722425` |
| `uv run llmzip compress x` | pass (expected) | `compress: not implemented yet`, exit 2 |

## 6. Smoke test results
The paragraph is 210 words, 1165 characters and 243 tokens. Speeds below are from the final run, after setting `NUM_THREADS=1`.

| Threads | Two runs bit-identical | Identical to threads=1 | 2 fresh processes match parent digest | tok/s |
|---|---|---|---|---|
| 1 | yes | yes | yes, yes | 179.3 |
| 4 | yes | yes | yes, yes | 150.1 |
| 10 (P-cores) | yes | yes | yes, yes | 128.2 |

- In the first run (with `NUM_THREADS=4`), the speeds were 158.0, 153.3 and 127.1 tok/s for 1, 4 and 10 threads. The ordering was the same both times.
- The logits digest was `8e19913f2b72243c…` for all thread counts and all processes.
- **Recommended NUM_THREADS: 1.** Single-token forward passes are small, so thread overhead outweighs the parallel speedup. It is now set in `config.py`.
- **Batched vs incremental** (information only): **not** bit-identical. The max |diff| was 6.714e-04 on the smoke paragraph and 3.662e-04 on the test sentence. This confirms why HARD RULE 3 exists.

**Ideal compressed size vs standard compressors:**

| Method | Bytes | Bits/char |
|---|---|---|
| raw | 1165 | 8.000 |
| ideal (sum of −log2 p) | 144.0 (1151.8 bits) | 0.989 |
| gzip -9 | 637 | 4.374 |
| xz -9 | 768 | 5.274 |
| zstd -19 | 638 | 4.381 |

The ideal size is about 4.4× smaller than gzip. The real coder will add a few bytes of arithmetic-coder termination, the cost of the frequency floor/quantization, and the container header. At this tiny size, header overhead also inflates the gzip, xz and zstd numbers.

## 7. Test results
`7 passed in 7.53s`. The tests are:
- the model loads;
- logits have shape (50257,) and dtype float32;
- two runs are bit-identical;
- the fingerprint is stable;
- the context-window guard raises;
- cross-process bit-identity (2 subprocesses, marked `slow`, runs by default);
- batched vs incremental (information only, never fails).

## 8. Deviations from the plan
- Additions beyond the original task spec, as you approved: cross-process determinism checks in the smoke script and the tests, fetching the revision sha from the Hub, and the batched-vs-incremental comparison.
- Also added: `xz -9` and `zstd -19` reference rows, a cross-thread-count identity check, a context-window guard test, and a `--fingerprint` flag on `info`, so plain `info` stays fast and doesn't load the model.
- transformers 5.x uses `dtype=` instead of the older `torch_dtype=` argument.
- The chip is an M4 **Pro** (10 performance cores), not a base M4.

## 9. Problems and open questions
- **No unresolved failures.**
- Every model load makes a network request to the HF Hub (the "unauthenticated requests" warning), even though the revision is pinned and the files are cached. Compression should probably work offline; see notes.
- Importing torch prints a harmless warning: `torch/distributed/elastic … Redirects are currently not supported in MacOs`.
- Logits matched across thread counts on this input. That isn't guaranteed in general, so the thread count stays in the fingerprint, and a file must be decompressed with the same `NUM_THREADS`.
- Bit-exactness across machines or torch builds is untested. For example, a different CPU, an Intel Mac or Linux might take different BLAS/vectorization paths. The fingerprint includes the versions but not the CPU type.

## 10. Notes for later phases
- **Phase 2/3:** use `NextTokenModel.start()` / `next_logits()` (or `logits_sequence`) on both sides. The decoder must call `start()` and then `next_logits(decoded_token)` in exactly the encoder's order.
- **Phase 3:**
  - Convert logits to probabilities deterministically. Softmax in float64 on CPU is deterministic, but the integer rounding must be identical on both sides.
  - With `FREQ_TOTAL_BITS=16`, 65536 total frequency is barely above the vocab size of 50257, so the floor of 1 per token eats about 77% of the mass. Strongly consider 24 or more bits; the coder's integer width must allow it.
- **Phase 3/4:** long inputs (more than 1023 tokens) need a strategy, such as chunking with a cache reset and re-BOS per 1023-token block, or a sliding window with re-encoding. The choice affects both the compression ratio and speed. Speed is about 180 tok/s, so a 100 KB text (~25k tokens) takes about 2.5 minutes.
- **Phase 4:**
  - Store the 16-hex fingerprint in the header, and refuse to decompress on a mismatch.
  - Consider setting `HF_HUB_OFFLINE=1` (or `local_files_only=True` once cached) so compression and decompression don't need the network.
  - Text detection needs to decide on tokenizer round-trip safety: does `decode(encode(text)) == text` byte-for-byte? It does for GPT-2's byte-level BPE on valid UTF-8, but this must be verified, with a fallback to zstd if it fails.
- **Phase 5:** time the per-token loop. Most of the cost is Python/HF overhead per call, which could be optimized later without breaking HARD RULE 3.

## 11. Git
- Commit: `0323486`
- Message: `Phase 1: project setup and model smoke test`
- The hash was written into this file after the commit, so this line is an uncommitted edit that goes into the Phase 2 commit.


# Phase 2 report: integer arithmetic coder

## 1. Summary
I implemented a Witten–Neal–Cleary binary arithmetic coder. It uses integers only, has 32-bit registers, and works with a frequency table that changes at every step. A `freqtable` helper module goes with it. There are 217 new tests, including a 50257-symbol, 2^24-total stress test that encodes long runs of freq-1 symbols; all pass. Compressed output is within 0.12% of the entropy. As housekeeping, the model now loads from the local cache with no network calls, and the fingerprint is unchanged.

## 2. Environment
Unchanged from Phase 1: macOS 15.7.1, Apple M4 Pro, uv 0.13.1, Python 3.12.15, xz 5.8.4, zstd 1.5.7, Apple gzip 457.140.3, git 2.50.1, torch 2.14.1, transformers 5.20.1. New direct dependency: **numpy 2.5.4**. It was already installed via transformers and is now declared in `pyproject.toml`, as you approved.

## 3. Files created/changed
| File | Purpose |
|---|---|
| `src/llmzip/config.py` | `FREQ_TOTAL_BITS = 24`, new `CODER_PRECISION_BITS = 32`, an import-time assert, and a comment explaining the invariant |
| `src/llmzip/coder.py` | Replaces the stub. WNC `Encoder`/`Decoder`, `BitWriter`/`BitReader` (MSB-first), interval validation, invariants documented in comments |
| `src/llmzip/freqtable.py` (new) | `validate`, `cumulative`, `symbol_interval` and `find_symbol` (via `np.searchsorted`); return values are Python ints |
| `src/llmzip/model.py` | Offline loading: sets `HF_HUB_OFFLINE=1` if the pinned snapshot is cached; `_from_pretrained` tries `local_files_only=True` first, then downloads |
| `src/llmzip/cli.py` | `info` imports `llmzip.model` first, so offline mode takes effect; it also prints the coder precision |
| `tests/test_coder.py` (new) | 217 coder and freqtable tests (see section 7) |
| `tests/test_model.py` | New `test_loads_without_network`: a fresh process with sockets blocked must load the model and produce the expected fingerprint |
| `scripts/demo_coder.py` (new) | Adaptive order-0 byte model plus the coder: round trip, sizes vs gzip, throughput |
| `pyproject.toml`, `uv.lock` | Add `numpy` |
| `CLAUDE.md` | Coder, freqtable and offline conventions; roadmap ticked |
| `README.md` | Status updated; demo command added |
| `reports/phase-1.md` | The commit-hash edit carried over from Phase 1 |
| `reports/phase-2.md` | This report |

## 4. Key decisions
- **Algorithm:** I used the Witten–Neal–Cleary (1987) arithmetic coder, with low/high registers, E1/E2 shifts and E3 underflow handled by a pending-bit counter. The register width is P = 32, so `HALF = 2^31` and `QUARTER = 2^30`.
- **Invariant:**
  - After renormalization, `low < HALF ≤ high`, and the interval doesn't sit inside the middle half. That gives `range = high − low + 1 > QUARTER`.
  - Since `total ≤ QUARTER`, `range ≥ total`, so every symbol with freq ≥ 1 gets a non-empty sub-interval.
  - `config.py` asserts `FREQ_TOTAL_BITS ≤ CODER_PRECISION_BITS − 2`, and `encode`/`consume` reject `total > 2^30`.
- **Narrowing:** `high = low + r*cum_high//total − 1` and `low = low + r*cum_low//total`. Python's integers are unbounded, so `r*cum` (up to 2^56) can't overflow.
- **Decoder target:** `((code − low + 1)·total − 1) // range`. This is the standard WNC formula, and it always lands in [0, total).
- **Termination:**
  - `finish()` emits 1 bit plus the pending bits: `01…` if `low < QUARTER`, else `10…`.
  - Followed by zeros, either pattern gives a value inside [low, high].
  - Trailing `0x00` bytes are then stripped, because the decoder supplies zero bits past the end.
  - **Zero symbols produce `b""`**, as a special case with a comment. The `Decoder` reads `code` lazily, so `Decoder(b"")` never reads anything when 0 symbols are requested.
- **Bit order:** bits are packed MSB-first within each byte, and bytes are written in order. The final partial byte is zero-padded.
- **Type safety:** the coder rejects anything that isn't a Python `int` with `TypeError`. numpy ints aren't `int` subclasses, so passing one by mistake fails loudly instead of risking silent int64 overflow.
- **freqtable:**
  - `cumulative(freqs, total)` returns an int64 array of length n+1 and validates that every freq is ≥ 1, the sum equals `total`, and `total ≤ 2^30`.
  - `find_symbol` uses `searchsorted(side="right") − 1`.
- **Offline loading:**
  - `HF_HUB_OFFLINE=1` is set with `setdefault`, so an explicit user setting still wins. It's only set if `<HF cache>/models--gpt2/snapshots/<MODEL_REVISION>/config.json` exists.
  - It has to be set before `huggingface_hub` is imported, which is why `cli.info` imports `llmzip.model` first.
  - As a second layer, `local_files_only=True` is tried first, with a download fallback on `OSError`.
- **Fingerprint:** unchanged (`bf3b4c5f19722425`). It doesn't include the coder or freq constants. Phase 4 should either store those in the container header or add them to the fingerprint.

## 5. Commands run and results
| Command | Result | Key output |
|---|---|---|
| `uv add numpy` | pass | numpy 2.5.4 declared (already installed) |
| `uv run pytest tests/test_coder.py` | pass | 217 passed in 11.13s |
| Mutation check (scratch copy, 3 deliberate coder bugs) | pass | each one caught: 207, 211 and 10 tests failed |
| `uv run python scripts/demo_coder.py README.md reports/phase-1.md` | pass (exit 0) | both round trips OK (see section 6) |
| `uv run pytest` | pass | 225 passed in 21.51s |
| `uv run llmzip info --fingerprint` | pass | `fingerprint: bf3b4c5f19722425`; no HF "unauthenticated requests" warning |
| Detector sanity check (`huggingface_hub.model_info` with sockets patched) | pass | prints `NETWORK ATTEMPT`, so the offline test's detector works |

## 6. Smoke test results (coder)
**Coder throughput.** These figures are coder-only: precomputed intervals replayed through `Encoder`/`Decoder`, at about 5 bits per symbol.

| File | Encode | Decode |
|---|---|---|
| README.md | 818k sym/s | 649k sym/s |
| phase-1.md | 769k sym/s | 586k sym/s |

- With the adaptive order-0 model, which builds a numpy cumsum and does a searchsorted per symbol, throughput is about 130–140k sym/s for encode and 120–126k sym/s for decode.
- For comparison, GPT-2 runs at about 180 tok/s, so the coder will be about 0.1–1% of runtime in Phase 3. The bottleneck will be the 50k-entry table construction plus the model.

**Efficiency vs entropy** (fixed 256-symbol table, total 2^24, 100,000 symbols):

| Top-symbol probability | Compressed | Entropy | Overhead |
|---|---|---|---|
| 0.999 | 217 B | 216.8 B | +0.115% |
| 0.9 | 15,825 B | 15,824.7 B | +0.002% |
| 0.5 | 62,451 B | 62,450.7 B | +0.000% |

The limit was 1% + 8 bytes. The overhead amounts to less than 1 byte in every case.

**Demo** (adaptive order-0 byte model, increment 32, halving when the total exceeds 2^16):

| File | Original | llmzip order-0 | Static order-0 entropy | gzip -9 |
|---|---|---|---|---|
| README.md | 845 B | 541 B (5.12 bits/byte) | 497.4 B | 502 B |
| reports/phase-1.md | 11,189 B | 7,105 B (5.08 bits/byte) | 7,045.1 B | 5,128 B |

Both round trips are byte-exact. Order-0 is expected to lose to gzip on longer text, because it doesn't model context. The demo only shows that the coder works with an adaptive, changing table.

## 7. Test results
`225 passed in 21.51s`: 217 coder tests and 8 model tests. The coder tests cover:
- a uniform 256-symbol table;
- a skewed table with the top symbol at 99.9%;
- an adaptive order-0 model (new table at every step);
- the LLM-like stress test: 50257 symbols, total 2^24, 1–7 heavy symbols redrawn every step, 1500 steps. Over 300 of the encoded symbols have freq exactly 1, including runs of 30 consecutive ones, plus indices 0 and 50256;
- edge cases: empty sequence (`b""`), single symbol, single-symbol alphabet, first and last index, and 10k runs of a common and of a rare symbol;
- 100k symbols;
- efficiency at three skews;
- 200 property seeds: alphabet 1–5000, random totals up to 2^24 that aren't powers of two, lengths 0–300, a table redrawn every step on half the seeds;
- determinism;
- invalid intervals and numpy-int rejection;
- freqtable validation and search;
- bit I/O order and zero-fill.

## 8. Deviations from the plan
- Extras added:
  - the efficiency test checks three skews instead of one;
  - a single-symbol-alphabet test (total = 1);
  - a long run of a rare symbol;
  - the bit-I/O unit test;
  - the `TypeError` for numpy ints;
  - a mutation check run by hand (not committed);
  - `info` prints the coder precision.
- Offline mode: `cli.info` had to import `llmzip.model` before `transformers`. Otherwise `HF_HUB_OFFLINE` is set too late, though `local_files_only=True` would still prevent network access.

## 9. Problems and open questions
- **No unresolved failures.**
- The `HF_HUB_OFFLINE` switch only works if `llmzip.model` is imported before `huggingface_hub`. Any future module that imports `transformers` first only gets the `local_files_only` protection. That's enough, as shown by the socket-blocked test, but worth keeping in mind.
- If someone sets `HF_HUB_OFFLINE=1` themselves while the snapshot is missing, the download fallback fails with an `OSError`. That's the expected behavior for an explicit user setting.
- `test_loads_without_network` hardcodes the fingerprint. Any intended change to torch, transformers or the thread count will require updating it. That's deliberate, because such a change also breaks compatibility with existing files.
- The torch distributed "Redirects" warning still appears on import. It's harmless.

## 10. Notes for later phases
- **Phase 3, probs:**
  - Build `freqs` as int64 numpy: every entry ≥ 1, summing to exactly 2^24. Run `freqtable.cumulative(freqs, 1 << FREQ_TOTAL_BITS)` once per step, and pass `int()` values to the coder.
  - The conversion must be bit-identical on both sides: same float64 softmax, same rounding, same tie-breaking when distributing the remainder. `np.argsort(kind="stable")` is a safe choice for tie-breaking.
- **Phase 3, performance:** per step, building a 50k cumsum (~50 µs) plus a searchsorted is negligible next to the ~5.5 ms forward pass. The coder itself takes about 1–2 µs per symbol.
- **Phase 4, header:** store the symbol (token) count, since the coder doesn't. Also store `FREQ_TOTAL_BITS` and `CODER_PRECISION_BITS`, or fold them into the fingerprint, so incompatible files are rejected.
- **Phase 4, CRC:** the coder strips trailing zero bytes. A truncated or corrupted stream decodes to garbage instead of raising, so the CRC32 is what detects corruption.
- **Phase 5, demo:** `scripts/demo_coder.py` could serve as a baseline row ("order-0 AC") in the benchmark.

## 11. Git
- Commit: `04d2962`
- Message: `Phase 2: integer arithmetic coder`
- The hash was written into this file after the commit, so this line is an uncommitted edit that goes into the Phase 3 commit.


# Phase 3 report: GPT-2 compression end to end

## 1. Summary
GPT-2 probabilities now drive the Phase 2 coder, giving lossless in-memory compression of texts up to 1023 tokens. Decompression in a separate process is verified by tests and by the demo. On the samples the output is 0.93–1.74 bits/char, 2.7–4.4× smaller than `gzip -9`, and the payload is within about 0.3% of the ideal model entropy. All 252 tests pass.

## 2. Environment
Unchanged from Phase 2: macOS 15.7.1, Apple M4 Pro, uv 0.13.1, Python 3.12.15, torch 2.14.1, transformers 5.20.1, numpy 2.5.4, xz 5.8.4, zstd 1.5.7, Apple gzip 457.140.3, git 2.50.1. No new dependencies.

## 3. Files created/changed
| File | Purpose |
|---|---|
| `src/llmzip/probs.py` | Replaces the stub: `softmax64` and `logits_to_freqs`, the only logits→freqs path |
| `src/llmzip/llmcodec.py` (new) | `compress_text` and `decompress_text`, the shared `_cum_tables` generator, `tokenize_checked`, the `NotRoundTrippable` and `TextTooLong` exceptions, and a `python -m llmzip.llmcodec` fresh-process helper |
| `src/llmzip/model.py` | Tokenizer flags (`add_special_tokens=False`, `split_special_tokens=True`, `skip_special_tokens=False`, `clean_up_tokenization_spaces=False`); the tokenizer's length warning is silenced |
| `tests/test_llmcodec.py` (new) | 27 tests (see section 7) |
| `scripts/demo_llm.py` (new) | Compresses each file in-process, decompresses it in a fresh process, and compares against gzip, xz and zstd |
| `bench/samples/prose.txt`, `prose_long.txt`, `sample.py`, `sample.json`, `sample.md` (new) | Original sample texts, committed |
| `CLAUDE.md`, `README.md` | Tokenizer, probs and codec conventions; status and demo command |
| `reports/phase-2.md` | Commit-hash edit carried over from Phase 2 |
| `reports/phase-3.md` | This report |

## 4. Key decisions
- **probs:** I kept your scheme unchanged.
  - p = float64 stable softmax; `freqs = floor(p·(2^24 − 50257)) + 1`.
  - The remainder goes to `argmax(p)`.
  - It asserts `0 ≤ remainder ≤ V`, `min ≥ 1` and `sum == 2^24`, and also checks that the logits are 1-D.
  - The remainder bound holds because `Σ floor(p·B) ∈ (B − V, B]`.
- **One shared path:**
  - `_cum_tables(model, n)` is a generator that calls `model.start()`, yields the cumulative table for step i, receives token i via `send()`, and then calls `next_logits(token)`.
  - The encoder sends the true token and the decoder sends the decoded one, so both sides run identical code.
  - The last token is never fed to the model, which is why exactly 1023 tokens fit (BOS + 1022 fed tokens = 1023 positions).
- **`<|endoftext|>` decision:** `split_special_tokens=True` tokenizes a literal `"<|endoftext|>"` as ordinary text (7 tokens), and it round-trips exactly. Token 50256 never appears inside the coded stream; it is only our BOS.
- **Round-trip safety:**
  - `tokenize_checked` encodes and decodes the text, and requires an exact match **before** any compression.
  - Any tokenizer exception, such as `TypeError` on a lone surrogate, becomes `NotRoundTrippable`.
  - The length check (`TextTooLong`) runs after the round-trip check.
- **Empty input:** `compress_text("")` returns `(b"", 0)`, and `decompress_text(_, 0)` returns `""` without loading the model.
- **Model reuse:** `get_model()` is a lazy process-wide singleton. Both functions also accept `model=`.
- **Fresh-process helper:** `python -m llmzip.llmcodec compress|decompress IN OUT`.
  - The payload file is a 4-byte big-endian `n_tokens` followed by the payload. This framing exists only for tests and demos; it is not the Phase 4 format.
  - Text is read and written as raw bytes, so CRLF is preserved.
  - It prints the coding time (excluding model load) as JSON.

## 5. Commands run and results
| Command | Result | Key output |
|---|---|---|
| In-process sanity round trip (smoke paragraph) | pass | 243 tokens → 144 bytes, decoded text equal |
| `uv run pytest tests/test_llmcodec.py -s` | pass | 27 passed in 49.69s; ideal 143.98 B, quantized 144.07 B, actual 144 B |
| `uv run python scripts/demo_llm.py bench/samples/*` | pass | 5/5 files byte-identical after fresh-process decompression |
| `uv run pytest` | pass | 252 passed in 68.64s |

## 6. Smoke test results
**Demo results.** Sizes are in bytes. Decompression ran in a fresh process for every file.

| File | OK | Original | Tokens | llmzip | Ideal | Quantized | Bits/char | gzip -9 | xz -9 | zstd -19 | Encode tok/s | Decode tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose.txt | ✅ | 1301 | 284 | **157** | 156.9 | 157.0 | 0.965 | 697 | 820 | 702 | 103.6 | 142.3 |
| prose_long.txt | ✅ | 4150 | 894 | **484** | 482.7 | 483.1 | 0.933 | 1909 | 2004 | 1872 | 101.8 | 110.8 |
| sample.json | ✅ | 775 | 358 | **109** | 108.5 | 108.7 | 1.125 | 365 | 428 | 368 | 134.7 | 150.3 |
| sample.md | ✅ | 653 | 196 | **113** | 112.1 | 112.1 | 1.391 | 424 | 508 | 427 | 129.1 | 147.1 |
| sample.py | ✅ | 761 | 309 | **165** | 166.9 | 164.8 | 1.735 | 444 | 540 | 452 | 131.7 | 126.5 |

**Ratio vs the best classic compressor:**

| File | llmzip vs best classic |
|---|---|
| prose_long | 3.9× smaller than zstd -19 |
| prose | 4.4× smaller than gzip -9 |
| json | 3.3× smaller |
| md | 3.8× smaller |
| py | 2.7× smaller |

On `prose_long` (4 KB), the classic compressors' headers no longer dominate, and the gap is still about 3.9×.

**Ideal vs quantized vs actual:**
- **Smoke paragraph:** ideal 143.98 B, quantized 144.07 B, actual 144 B.
- **Cost of quantization:** about 0.1 B per text, under 0.1%.
- **Coder overhead:** under 1 B per text.
- `sample.py` comes out *below* ideal (165 vs 166.9). This happens because the argmax token receives the remainder mass (up to about 0.3%); in code the actual token is very often the argmax, so it gains slightly.

**Throughput:** encode runs at about 100–135 tok/s and decode at about 110–150 tok/s.
- These rates are below Phase 1's bare-model 180 tok/s. The extra per-token cost is the float64 softmax over 50257 entries, the cumsum and the coder, plus attention cost that grows with context. On the longer file, decode drops to 111 tok/s.
- The first file's encode also includes warm-up.

**Cross-process round trip:** ✅ in every case.
- The tests compress in subprocess A, write a temp file, and decompress in subprocess B. Both the smoke paragraph and an edge mix (CRLF, NUL, emoji, `<|endoftext|>`, a combining accent) came back byte-identical.
- The fresh-process freq-table SHA-256 also matches the parent's.
- The demo decompressed all 5 samples in fresh processes, all byte-identical.

## 7. Test results
`252 passed in 68.64s`: 27 new codec tests, 217 coder tests and 8 model tests. The new tests are:
- **probs:** invariants (sum exactly 2^24, every freq ≥ 1, int64, the argmax gets exactly the remainder, all other entries equal floor+1), determinism, a bad-shape rejection, and a SHA-256 of the freq arrays matching in a fresh process.
- **Round trips:** the smoke paragraph, Python, JSON, markdown, a single character and a single space.
- **11 tokenizer edge cases:** CRLF, tabs, trailing whitespace, multiple blank lines, emoji (including a ZWJ sequence), CJK, a combining accent (`é`), NUL, leading spaces, a literal `<|endoftext|>` (also asserting that 50256 is not among the ids) and `" , . !"`-style punctuation.
- **Boundaries:** a text of exactly 1023 tokens round-trips, and 1024 tokens raises `TextTooLong`.
- **Other:** the empty string; a lone surrogate raises `NotRoundTrippable`; efficiency within 1% + 8 B of ideal; 2 cross-process round trips.

## 8. Deviations from the plan
- Additions you approved: the 1023 and 1024 boundary tests, and `bench/samples/prose_long.txt` (894 tokens).
- Small additions:
  - a single-space round trip;
  - a ZWJ emoji in the emoji case;
  - a shape check in `logits_to_freqs`;
  - the demo also prints the ideal and quantized sizes;
  - I silenced the tokenizer's "sequence length > 1024" warning (`model_max_length`), since limits are enforced per context.
- `decompress_text` with `n_tokens == 0` returns `""` without loading the model.

## 9. Problems and open questions
- **No unresolved failures.**
- **Runtime:** the 1023-token round-trip test takes about 23 s, about 11 ms per token across both directions. Attention cost grows with the KV cache length, so the full suite now takes about 69 s.
- **Cross-platform determinism:** numpy's `exp` (SIMD) and torch's kernels are deterministic on this machine and these builds, but they are not guaranteed identical across numpy versions, torch versions or CPUs. The current fingerprint covers torch and transformers but **not numpy**.
- `sample.py` came out below ideal, as explained in section 6. It is not a bug, just a side effect of giving the remainder to the argmax.

## 10. Notes for later phases
- **Phase 4, fingerprint/header:**
  - Add the numpy version to the fingerprint, or store it in the header. Store `FREQ_TOTAL_BITS` and `CODER_PRECISION_BITS` too.
  - Changing the fingerprint makes Phase 1 and 2 files incompatible. None exist yet, so this costs nothing now.
  - `test_loads_without_network` hardcodes the fingerprint and will need updating.
- **Phase 4, chunking:**
  - Simplest: split the token ids into blocks of at most 1023, and reset with a fresh BOS for each block. Store each block's token count, or a total plus a fixed block size.
  - Split on token ids, not text. Tokenize once, round-trip-check the whole text, then cut the id list. Decoding the concatenated ids once at the end avoids UTF-8 sequences split across a block boundary.
  - Expect a small loss at each block start, because the model loses its context.
- **Phase 4, routing:** `NotRoundTrippable` and `UnicodeDecodeError` (non-UTF-8 input) should route to the zstd fallback. Also consider falling back when the LLM payload would be larger than the zstd output.
- **Phase 4, CRC:** the coder can't detect corruption, so the CRC32 over the original bytes is essential.
- **Phase 5, benchmarks:**
  - Speed is about 110–140 tok/s, so 100 KB of text (about 25k tokens) takes about 3–4 minutes each way.
  - Possible speedups that keep HARD RULE 3:
    - compute the softmax in torch float64 instead of numpy;
    - avoid `np.floor` on all 50k entries;
    - fuse the cumsum.
  - These must not change the results. Verify any speedup with the fresh-process freq-hash test.
- **Phase 6, inspect:** `softmax64(logits)[token]` per step gives the per-token surprise directly. Reuse the `_cum_tables` path, or `logits_sequence` for display only.

## 11. Git
- Commit: `74084f6`
- Message: `Phase 3: GPT-2 compression end to end`
- The hash was written into this file after the commit, so this line is an uncommitted edit that goes into the Phase 4 commit.


