# Phase 4 report: file format, fallback, chunking and CLI

## 1. Summary
`llmzip compress` and `llmzip decompress` now work on any file of any size, in separate processes. Each file is written in a versioned, checksummed container (STORED, ZSTD or LLM), and "never worse" routing picks the smallest. Text longer than one context is coded in independent blocks of 1023 tokens.

Corruption and incompatibility raise clean errors (exit 1) and never leave a partial output file. All 8 demo inputs came back byte-identical, including a 10,272-token (11-block) markdown file, random bytes and a PNG. All 309 tests pass.

## 2. Environment
Unchanged: macOS 15.7.1, Apple M4 Pro, uv 0.13.1, Python 3.12.15, torch 2.14.1, transformers 5.20.1, numpy 2.5.4, zstandard 0.25.0, rich 15.0.0, xz 5.8.4, zstd 1.5.7, Apple gzip 457.140.3, git 2.50.1. No new dependencies.

## 3. Files created/changed
| File | Purpose |
|---|---|
| `src/llmzip/errors.py` (new) | `LlmzipError` → `ContainerError` → `BadMagic`, `UnsupportedVersion`, `Truncated`, `CorruptContainer`, `FingerprintMismatch`, `ChecksumMismatch` |
| `src/llmzip/container.py` | Replaces the stub: the v1 format, `pack`/`unpack` with sanity checks, uvarint (LEB128), `verify` (size + CRC32), `check_fingerprint` |
| `src/llmzip/fallback.py` | Replaces the stub: zstd level 19 compress/decompress; zstd errors become `CorruptContainer` |
| `src/llmzip/api.py` (new) | `compress_bytes` (mode routing, "never worse") and `decompress_bytes` (fingerprint check before decoding, then verify) |
| `src/llmzip/llmcodec.py` | Chunking: `compress_text`/`decompress_text` work on block lists, with a single detokenize at the end and a per-token progress callback. `TextTooLong` and the Phase 3 `__main__` helper are removed |
| `src/llmzip/model.py` | Wider fingerprint, `fingerprint_bytes()`, `ENCODE_KWARGS`/`DECODE_KWARGS` as the single source of tokenizer flags; the torch "Redirects" warning and the HF "Loading weights" bar are silenced |
| `src/llmzip/probs.py` | `PROBS_SCHEME = "floor+1, remainder-to-argmax, v1"` |
| `src/llmzip/config.py` | `MAX_BLOCK_TOKENS = 1023` |
| `src/llmzip/cli.py` | Real `compress` and `decompress`, atomic writes, rich progress bar, summary line, exit codes, `LLMZIP_DEBUG`; `info` shows the new fields |
| `tests/helpers.py` (new) | Shared `PARAGRAPH`, `SAMPLES`, `text_with_exactly`, `run_cli` |
| `tests/test_container.py` (new) | 36 format, uvarint and corruption tests |
| `tests/test_cli.py` (new) | 20 routing, corruption, chunking and subprocess CLI tests |
| `tests/test_llmcodec.py` | Adapted to the blocks API; the 1024-token test is now a 2-block round trip; cross-process tests go through the CLI |
| `tests/test_model.py` | `EXPECTED_FINGERPRINT` updated |
| `scripts/demo_cli.py` (new) | CLI end to end: compress, decompress in a separate process, `cmp`; finds a PNG |
| `scripts/demo_llm.py` | Blocks API, per-block ideal size, fresh-process decompression via the CLI |
| `bench/samples/reports_concat.md` (new) | `reports/phase-1..3.md` concatenated: 32,883 B, 10,272 tokens |
| `CLAUDE.md`, `README.md` | Format, routing, errors, fingerprint rules; real usage |
| `reports/phase-3.md` | Commit-hash edit carried over from Phase 3 |
| `reports/phase-4.md` | This report |

## 4. Key decisions
- **Fingerprint:** the new value is **`2bb9a6826f48e26a`** (it was `bf3b4c5f19722425`).
  - Added inputs: the numpy version, `FREQ_TOTAL_BITS`, `CODER_PRECISION_BITS`, the sorted tokenizer kwargs and `PROBS_SCHEME`.
  - The tokenizer kwargs are now module constants used by both `encode`/`decode` and the fingerprint, so they can't drift.
  - The 8 raw bytes in the header are `bytes.fromhex` of the 16-hex fingerprint.
- **Container:** I implemented the layout exactly as you specified. The header is 18 bytes.
  - The LLM body adds 22 bytes plus a 1–2 byte uvarint per block, so LLM-mode overhead is about 41–42 bytes for a single block.
  - **`unpack` sanity checks**, all run before any decoding:
    - `1 ≤ block_size ≤ 1023`;
    - `n_blocks == ceil(n_tokens / block_size)`;
    - `n_tokens ≤ original_size` (every GPT-2 token is at least 1 byte);
    - `n_tokens == 0` exactly when the size is 0;
    - the sum of block lengths equals the remaining bytes (short gives `Truncated`, extra gives `CorruptContainer`).
  - A STORED body must be exactly `original_size` bytes.
  - With these checks, a corrupted count can't trigger a huge decode.
- **Routing (`api.compress_bytes`):** STORED is always a candidate; ZSTD is one if the data is non-empty; LLM is one if the data is valid UTF-8 and round-trippable.
  - The smallest *packed container* wins, and ties go to STORED, then ZSTD, then LLM.
  - Because the header is shared, this is the same as your rules 4 and 5.
  - Forced `--mode llm` raises on non-UTF-8 or non-round-trippable input. A forced mode on empty input gives STORED.
- **Strict UTF-8:** every conversion uses `.decode("utf-8")` and `.encode("utf-8")`, never `utf-8-sig`. A BOM becomes U+FEFF in the text, goes through GPT-2 like any other character, and comes back byte-identical (tested).
- **Chunking:**
  - The whole text is tokenized and round-trip checked once. The ids are cut into `block_size` slices, and each block restarts with `model.start()` through the same `_cum_tables` generator.
  - Decode collects all ids, then detokenizes once. A test with `block_size` 1, 2, 3 and 7 on emoji and CJK cuts through multi-byte characters and still round-trips.
- **Fingerprint before decoding:** `decompress_bytes` loads the model, compares fingerprints, and only then decodes. A spy model whose `start` and `next_logits` raise proves no model computation happens on a mismatch.
- **CLI safety:**
  - Overwrite and same-file checks run before any work.
  - Output is written to a temp file in the destination directory, fsynced, then moved into place with `os.replace`. The temp file is deleted on any exception.
  - Decompression verifies size and CRC32 *in memory* before writing, so a mismatch never creates an output file at all.
  - `--verify` decompresses in-process before writing.
- **Errors:**
  - `LlmzipError`, `OSError` and `ValueError` become `llmzip: error: …` with exit 1.
  - A bad output name is `llmzip: usage error: …` with exit 2 (argparse errors also exit 2).
  - `LLMZIP_DEBUG=1` re-raises with the full traceback.
- **Progress bar:** rich on stderr, showing the label, a bar, `done/total tok`, tok/s and ETA. It's enabled only when stderr is a TTY and `--quiet` isn't set, and it's transient.
- **CLI noise:** I silenced the torch "Redirects are currently not supported in MacOs" warning (logger level) and transformers' "Loading weights" bar, so CLI output is clean.

## 5. Commands run and results
| Command | Result | Key output |
|---|---|---|
| `uv run llmzip info --fingerprint` | pass | `fingerprint: 2bb9a6826f48e26a`, plus probs scheme, numpy and max block tokens |
| Manual CLI session (scratchpad) | pass | `compress --verify` → `mode=LLM 1301 -> 199 B (ratio 6.54x, 1.224 bits/char, verified)`; decompress in a new process, `cmp` OK; overwrite refused (exit 1); non-`.lz` name (exit 2); random 4 KB → STORED; empty file with `--mode zstd` → STORED 18 B round trip |
| `uv run pytest tests/test_container.py` | pass | 36 passed in 0.02s |
| `uv run pytest tests/test_cli.py tests/test_llmcodec.py tests/test_model.py` | pass | 56 passed in 120.59s |
| `uv run python scripts/demo_cli.py` | pass (exit 0) | 8/8 `cmp` OK (see section 6), 4 min 35 s total |
| Progress bar, forced terminal rendering (`TTY_COMPATIBLE=1`) | pass | frame: `compress ━━╸━━… 27/196 tok 172 tok/s 0:00:01` |
| `LLMZIP_DEBUG=1 llmzip decompress bad.lz` vs without | pass | traceback with `UnsupportedVersion` vs a one-line `llmzip: error: unsupported format version 9 (this build reads 1)`, exit 1 |
| `uv run pytest` | pass | 309 passed in 131.34s |
| `uv run python scripts/demo_llm.py` (5 small samples, run after the commit) | pass (exit 0) | all OK; payload sizes identical to Phase 3 (157/484/109/113/165 B), so the refactor changed no coded bits; the `.lz` sizes add 41–42 B |

## 6. Smoke test results (CLI demo)
Each file was compressed by one `llmzip` process and decompressed by a separate one, then compared with `cmp`.

| File | Mode | Original | .lz | Ratio | gzip -9 | Compress | Decompress | cmp |
|---|---|---|---|---|---|---|---|---|
| prose.txt | LLM | 1,301 | 199 | 6.54× | 707 | 4.1 s | 3.9 s | ✅ |
| prose_long.txt | LLM | 4,150 | 526 | 7.89× | 1,924 | 11.2 s | 11.8 s | ✅ |
| reports_concat.md (10,272 tok, 11 blocks) | LLM | 32,883 | 7,419 | 4.43× | 12,781 | 106.1 s | 105.5 s | ✅ |
| sample.json | LLM | 775 | 150 | 5.17× | 377 | 5.3 s | 4.8 s | ✅ |
| sample.md | LLM | 653 | 154 | 4.24× | 434 | 3.4 s | 3.4 s | ✅ |
| sample.py | LLM | 761 | 207 | 3.68× | 454 | 4.4 s | 4.2 s | ✅ |
| random 64 KiB (`$TMPDIR/llmzip_random_64k.bin`) | STORED | 65,536 | 65,554 | 1.00× | 65,596 | 1.8 s | 1.8 s | ✅ |
| PNG (`/System/Library/Desktop Pictures/.wallpapers/Sequoia Sunrise/Sequoia Sunrise Thumbnail.png`, the preferred path existed) | ZSTD | 70,450 | 69,837 | 1.01× | 69,902 | 1.8 s | 1.9 s | ✅ |

- **Times** include process start, the about 2 s model load and the about 1 s weight hash for the fingerprint. The binary files never load the model.
- **gzip sizes** here include the file name stored in the gzip header (`gzip -9 -c FILE`), so they are slightly larger than Phase 3's piped numbers.
- **Throughput:** the 11-block file ran at about 97 tok/s each way including startup, so about 100 tok/s steady state.
- **reports_concat.md compresses less well** (1.80 bits/byte) than prose (0.93–1.2) because it's full of tables, hashes and numbers. It still beats gzip -9 by 1.7×.
- **"Never worse" holds:** random bytes were stored (zstd would have been larger), and the PNG took zstd's 0.9% gain over STORED.
- **Container overhead:** compared with Phase 3's bare payloads, LLM files gain 42 B (18 header + 22 LLM header + 2-byte uvarint). For example, prose.txt's 157 B payload becomes a 199 B file.

## 7. Test results
`309 passed in 131.34s`: 36 new container tests, 20 new CLI/API tests, 28 codec tests, 217 coder tests and 8 model tests.
- **Container:**
  - round trips for all 3 modes, and the little-endian layout;
  - uvarint at 0, 1, 127, 128, 255, 16383, 16384, 2^32, 2^63 and 2^64−1, known encodings, truncated, overlong (11 bytes) and above-u64 values;
  - bad magic (including an empty file), unsupported version, truncated header at 5, 10 and 17 bytes for every mode, truncated body, truncated LLM header;
  - trailing data, unknown mode, inconsistent block count, block size 0 or 1024;
  - an n_tokens of 2^56 rejected before decoding;
  - a flipped byte in a STORED or ZSTD body, a flipped CRC field, and the fingerprint check.
- **Routing:**
  - empty → STORED (18 B);
  - random 4 KB → ZSTD or STORED;
  - invalid UTF-8 → ZSTD (no LLM candidate);
  - `"hello"` → STORED (the smallest);
  - prose → LLM;
  - a UTF-8 BOM plus prose → LLM, byte-identical;
  - not round-trippable → ZSTD (forced with a monkeypatch, see section 9);
  - each forced mode, and forced `llm` on invalid UTF-8 → error.
- **Corruption via the API:** a flipped LLM payload byte gives a `ContainerError`; a modified fingerprint gives `FingerprintMismatch` with the spy model never run.
- **Chunking:**
  - `block_size=50` on prose.txt (284 tokens: 6 blocks, last one partial);
  - exactly 300 tokens (6 full blocks);
  - multi-byte splits at block sizes 1, 2, 3 and 7;
  - the container's block fields.
- **Subprocess CLI:**
  - a real 2-block round trip (prose_long + prose, about 1180 tokens) in separate processes;
  - default names plus `--verify`;
  - overwrite refusal and `--force`, with the input untouched;
  - a non-`.lz` name without `-o` exits 2;
  - a corrupt file exits 1 with a clean message and no output or temp file;
  - bad magic exits 1.
- **Codec:** a 1024-token round trip as 2 blocks (the first block is exactly 1023 tokens, a full context) and `block_size` out of range.

## 8. Deviations from the plan
- **1023/1024 tests:** I replaced Phase 3's separate "1023 round trip" and "1024 raises" tests with one 1024-token round trip, asserting 2 blocks. Its first block is exactly 1023 tokens, so the full-context case is still covered, and it saves about 20 s of test time.
- **Forced mode on empty input:** `--mode zstd|llm` on an empty input produces STORED. An empty input has nothing to compress, and the result is the 18-byte minimum.
- **`NotRoundTrippable` routing test:** this uses a monkeypatch. Valid UTF-8 bytes always decode to a `str` without lone surrogates, and I found no natural input where GPT-2's byte-level tokenizer fails to round-trip one. So the ZSTD fallback for this case is real code, but unreachable in practice so far.
- **Progress bar:** the sandbox couldn't allocate a pty (`script` failed), so I verified rendering with rich's forced-terminal mode instead of a real TTY.
- **Small additions:**
  - `--block-size` (used by tests; also handy for experiments);
  - `-q` as a short form of `--quiet`;
  - `info` prints numpy, the probs scheme and max block tokens;
  - silenced the torch and HF stderr noise.
- **Demo files:** the random file went to the system temp dir (`$TMPDIR`) instead of the scratchpad. It isn't committed either way.

## 9. Problems and open questions
- **No unresolved failures.**
- **Auto mode always runs the full LLM encode on text,** even when zstd would obviously win (for example, a large log file). At about 100 tok/s, 1 MB of text (about 250k tokens) would take about 40 minutes. Estimating on a prefix first could skip hopeless LLM attempts; see section 10.
- **Fingerprint cost:** hashing the 500 MB of weights takes about 1 s on every LLM compress or decompress. This could be cached, keyed on the safetensors file's path, size and mtime.
- **Late corruption detection:** a corrupted LLM payload is only detected after a full decode, because the CRC covers the output. Detection is guaranteed, but it costs a full decode.
- **Memory:** the whole file is held in memory, which is fine at today's speeds.
- **Progress bar:** not tested under a real pty in this environment.

## 10. Notes for later phases
- **Phase 5, bench:**
  - Build `llmzip bench` on `scripts/demo_cli.py`'s pattern, and use the piped `gzip -9 -c < file` so the comparison isn't inflated by the stored file name.
  - Measure the block-restart cost: compress `reports_concat.md` with block sizes 1023, 512 and 256 to quantify it.
  - Report bits/char, tok/s and wall time including model load separately.
- **Phase 5, performance:**
  - An auto-mode heuristic: if zstd's ratio on the whole file beats the LLM's ratio on the first block by a wide margin, skip the LLM.
  - Cache the weights hash.
  - Speed up probs (torch float64 softmax, fused cumsum). Any change that can alter frequencies must bump `PROBS_SCHEME`, and must be verified with the fresh-process freq-hash test.
- **Future, context:** sliding-window or overlapping context would avoid the cold start at each block. That needs a format change (format version 2) and a matching decoder, and it remains incremental under HARD RULE 3.
- **Phase 6, inspect:** reuse `llmcodec`'s block structure (restart per block) so the heatmap matches what is actually coded. `softmax64(logits)[token]` gives the per-token surprise.

## 11. Git
- Commit: `8601f1d`
- Message: `Phase 4: file format, fallback, chunking and CLI`
- The hash was written into this file after the commit, so this line is an uncommitted edit that goes into the Phase 5 commit.
