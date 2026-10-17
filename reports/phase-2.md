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
