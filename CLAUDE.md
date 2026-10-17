# llmzip

`llmzip` is a command-line tool that losslessly compresses text files using a small local language model (GPT-2) combined with arithmetic coding.

- **Compression:** tokenize the text. For each token, get the model's next-token distribution given all previous tokens, convert it to integer frequencies, and arithmetic-encode the actual token.
- **Decompression:** run the identical model with identical settings and decode token by token.
- **Fallback:** non-text files fall back to zstd. The header stores the mode, model fingerprint, original length and CRC32.

## HARD RULES

1. CPU only. Never use MPS/Metal/CUDA. Set a fixed thread count via
   `torch.set_num_threads(N)` where N comes from a single config constant.
2. Call `torch.use_deterministic_algorithms(True)` and run the model in
   `torch.no_grad()` and eval mode, float32.
3. The compressor and decompressor MUST obtain probabilities through ONE shared
   function that processes tokens one at a time (incremental, with KV cache).
   Never compute encoder probabilities with a single batched forward pass over the
   whole sequence, even though it would be faster — batched and incremental results
   can differ numerically and that silently corrupts decompression.
4. The arithmetic coder is integer-only. No floats inside the coder.
5. Probabilities are converted to integer frequency tables before coding; every
   token gets a frequency of at least 1.
6. Every change to coding logic must be followed by running round-trip tests
   (compress → decompress → byte-for-byte equality).
7. Stay within the current phase's scope. If you notice something for a later
   phase, write it under "Notes for later phases" in the report instead of building it.
8. Keep dependencies minimal: torch, transformers, zstandard, rich, pytest.
   Ask before adding anything else.
9. Commit to git at the end of each phase with message "Phase N: <summary>".

## Conventions

- Approved dependency additions beyond HARD RULE 8's list: numpy (phase 2) and textual (phase 7).

- All settings live in `src/llmzip/config.py`: `MODEL_NAME`, the pinned `MODEL_REVISION` (HF commit sha), `NUM_THREADS`, `ATTN_IMPLEMENTATION="eager"`, `BOS_TOKEN_ID` and `FREQ_TOTAL_BITS`.
- **Shared probability source:** `NextTokenModel` in `src/llmzip/model.py`.
  - `start()` feeds `<|endoftext|>` (50256) as BOS and returns the logits for the first real token.
  - `next_logits(t)` feeds one token and returns the logits for the next one.
  - `logits_sequence(ids)` wraps both: it returns one logits tensor per token in `ids`.
- The context is limited to 1024 positions, including BOS. `next_logits` raises when the window is full.
- `model_fingerprint()` is 16 hex characters of SHA-256 over: model name, revision, weights SHA-256, torch and transformers versions, thread count and attention implementation. It goes in file headers (phase 4).
- Measured in phase 1: batched logits differ from incremental ones (max |diff| ~7e-4). Incremental logits are bit-identical across runs, across thread counts 1/4/10 and across fresh processes.
- **Coder** (`src/llmzip/coder.py`): a Witten–Neal–Cleary binary arithmetic coder using 32-bit registers (`CODER_PRECISION_BITS`). It is integer-only.
  - `Encoder.encode(cum_low, cum_high, total)` and `Encoder.finish() -> bytes`.
  - `Decoder(data).get_target(total)`, then `Decoder.consume(cum_low, cum_high, total)` with the same interval.
  - The table may change at every step. `total` must be ≤ 2^30; we use 2^24 (`FREQ_TOTAL_BITS`).
  - The coder doesn't store the symbol count; the caller must.
  - Bits are packed MSB-first. Trailing zero bytes are stripped, and the decoder zero-fills past the end of the data. Encoding zero symbols gives `b""`.
  - Arguments must be Python `int`, never numpy ints; anything else raises `TypeError`.
- **Freq tables** (`src/llmzip/freqtable.py`): `cumulative(freqs, total)` builds an int64 array of length n+1 and validates it (every freq ≥ 1, the sum equals total). `symbol_interval` and `find_symbol` return Python ints.
- **Tokenizer** (`NextTokenModel.encode`/`decode`): encoding uses `add_special_tokens=False, split_special_tokens=True`, and decoding uses `skip_special_tokens=False, clean_up_tokenization_spaces=False`.
  - A literal `"<|endoftext|>"` in the input is tokenized as ordinary text, so token 50256 only ever appears as our own BOS.
- **probs** (`src/llmzip/probs.py`): `logits_to_freqs` is the ONLY logits→freqs path.
  - p = float64 stable softmax; freqs = floor(p·(2^24 − V)) + 1.
  - The remainder goes to `argmax(p)`, taking the first index on ties.
- **Phase 5 constraint, still in force:** speed or tooling changes must not alter coded bits. Verify with a byte comparison of `.lz` files against the previous phase.
- **Codec** (`src/llmzip/llmcodec.py`):
  - `compress_text(text, model, block_size=1023) -> (blocks, n_tokens)`; `decompress_text(blocks, n_tokens, model, block_size) -> str`.
  - The whole text is tokenized and round-trip checked once, then the ids are cut into blocks. Each block restarts from `model.start()` (fresh BOS, empty cache).
  - Decoding collects all ids and detokenizes ONCE, so multi-byte characters split across blocks are reassembled.
  - Everything goes through the single `_cum_tables` generator. `NotRoundTrippable` is raised if decode(encode(text)) != text.
- **Container v1** (`src/llmzip/container.py`, little-endian):
  - Header: `LLMZ` | version u8 = 1 | mode u8 (0 STORED, 1 ZSTD, 2 LLM) | original_size u64 | crc32 u32 of the original bytes. That's 18 bytes.
  - LLM body: fingerprint 8s | block_size u16 | n_tokens u64 | n_blocks u32 | n_blocks × uvarint block lengths | payloads.
  - `unpack` validates the structure and sanity-checks the counts before any decoding. `verify` checks the size and CRC after decoding.
  - Errors live in `src/llmzip/errors.py` (`LlmzipError` → `ContainerError` → `BadMagic`, `UnsupportedVersion`, `Truncated`, `CorruptContainer`, `FingerprintMismatch`, `ChecksumMismatch`).
- **Routing** (`src/llmzip/api.py`, mode=auto, "never worse"):
  - Empty input → STORED.
  - Invalid UTF-8 or not round-trippable → candidates STORED and ZSTD.
  - Otherwise → STORED, ZSTD (level 19) and LLM.
  - The smallest container wins; ties go to STORED, then ZSTD, then LLM.
  - Bytes↔text conversion is always strict `"utf-8"`, never `utf-8-sig`, so a BOM survives as U+FEFF.
  - Decompress checks the fingerprint BEFORE running the model.
  - **Skip heuristic:** in auto mode, for text longer than 2 blocks, the first block is coded alone. The full LLM payload is estimated from its bits/byte; if that exceeds the zstd frame, the LLM is skipped.
    - Otherwise the first block is reused via `compress_text(done_blocks=...)`, so the output is bit-identical to a no-skip run.
    - `--no-skip` disables the heuristic; `api.skip_estimate` is shared with bench.
- **Fingerprint inputs:** model name, revision, weights SHA-256, the torch, transformers and numpy versions, threads, attention implementation, `FREQ_TOTAL_BITS`, `CODER_PRECISION_BITS`, the tokenizer kwargs (`ENCODE_KWARGS`/`DECODE_KWARGS`) and `probs.PROBS_SCHEME`. **Bump `PROBS_SCHEME` whenever probs.py's output can change.** Update `EXPECTED_FINGERPRINT` in `tests/test_model.py` deliberately.
- **Weights-hash cache:** `~/Library/Caches/llmzip/weights_sha256.json` (or `$LLMZIP_CACHE_DIR`).
  - Keyed by the resolved `model.safetensors` path, size, mtime_ns and `MODEL_REVISION`.
  - Any cache problem means a recompute; the cache must never change the fingerprint.
- **LLM token budget:** `config.LLM_TOKEN_BUDGET = 25000` and `compress --max-llm-tokens N` (0 = unlimited). Auto mode checks it right after tokenizing, before any model work, and then uses ZSTD or STORED. Forced `--mode llm` ignores it.
- **Inspect** (`src/llmzip/inspect_cmd.py`) is read-only with respect to compression.
  - It uses `tokenize_checked`, the same per-block `model.start()` restarts, and `probs.softmax64` / `probs.logits_to_freqs`.
  - It codes each block in the same pass with the real `Encoder`. Tests assert those bytes equal `compress_text`'s output, so keep it that way if either side changes.
  - Token text segments come from GPT-2's byte↔unicode table, with multi-byte characters assigned to the token that completes them.
- **TUI** (`src/llmzip/tui/`, Textual, `llmzip tui [PATH]`) is a front end only.
  - It calls `api.compress_bytes` / `api.decompress_bytes`, `inspect_cmd.analyze(top_k=5)` and `bench.bench_file`, and never re-implements coding, probs or the format.
  - It shares `fileops` (output naming, `check_output`, `write_atomic`) and `api.skip_notes` with the CLI. A test asserts that TUI and CLI `.lz` output are byte-identical.
  - **Workers:** every slow task is a thread worker; the model loads once at startup, and jobs run in group "job", one at a time, with buttons disabled while busy. Workers touch widgets only via `call_from_thread`.
  - **Cancellation:** the Cancel key sets a `threading.Event`, and the api progress callback raises `errors.Cancelled` on the next token. Output is written (atomically) only after a job finishes, so cancelled or failed jobs write nothing.
  - **Errors** become notifications, never crashes; with `LLMZIP_DEBUG=1` tracebacks go to `cache_dir()/tui-error.log`.
  - **Heatmap** (`tui/heatmap.py`) uses the line API with soft-wrap. Spans are recorded per visual line for exact clicks, and colours come from `inspect_cmd.BUCKETS`. `scrollbar-gutter: stable` keeps the wrap width equal to the drawn width.
  - **Tests** (`tests/test_tui.py`) run headless via `App.run_test()` inside `asyncio.run()`, with an injected model. Wait on worker groups `job`/`info`/`model` only, because DirectoryTree's own `_loader` worker never finishes.
- **Restored copies** (`fileops`):
  - `restored_output` inserts `.restored` before the last extension (`report.md.lz` → `report.restored.md`).
  - `unique_path(where="base")` is for `.lz` outputs (`report-2.md.lz`); `where="last"` is for restored copies (`report.restored-2.md`).
  - `identity_check` compares a restored copy with the original next to it (the `.lz` name minus `.lz`) using size and SHA-256.
  - Used by `decompress --restored/--unique`, `compress --unique`, the TUI "Restore as copy" switch (on by default) and the Finder scripts.
- **Finder Quick Actions** (`finder/`): zsh scripts plus `common.sh`.
  - They call the CLI only (through `uv run --project REPO --frozen`), so there's no naming or coding logic in shell.
  - `install.sh` generates `.workflow` bundles with `make_workflow.py` (plistlib) and bakes in `LLMZIP_UV` and `LLMZIP_REPO`, because Automator has a minimal PATH. It runs `plutil -lint` on them.
  - Test hooks: `LLMZIP_NOTIFY_CMD`, `LLMZIP_LOG_FILE`, `LLMZIP_DRY_RUN`, and `install.sh --dest DIR`. Never run `install.sh` without `--dest` in tests.
- **CLI:**
  - `llmzip compress IN [-o OUT] [--mode auto|llm|zstd|store] [--block-size N] [--force] [--verify] [-q]`.
  - `llmzip decompress IN [-o OUT] [--force] [-q]`.
  - `llmzip bench FILE... [--block-sizes 1023,512] [--no-decompress] [--out bench/results]`. New results are gitignored; publish one with `git add -f`.
  - `llmzip inspect FILE [--top N] [--block-size N] [--max-tokens N] [--json OUT] [--no-color]`.
  - `llmzip tui [PATH] [--max-tokens N] [--tab overview|heatmap|stats|compare]` (`t` toggles the tree; dotfiles are hidden).
  - `llmzip decompress --restored [--unique]` and `llmzip compress --unique`.
  - Writes are atomic (temp file + `os.replace`). Exit codes: 0 ok, 1 error, 2 usage. `LLMZIP_DEBUG=1` shows tracebacks.
- **Offline:** `model.py` sets `HF_HUB_OFFLINE=1` when the pinned snapshot is already cached, and loads with `local_files_only=True` first. Import `llmzip.model` before `transformers`.
- Each phase writes a report to `reports/phase-N.md`.

## Phase roadmap

1. Setup, deterministic incremental model, smoke test. ✅
2. Integer-only arithmetic coder (`coder.py`) plus round-trip tests on synthetic distributions. ✅
3. Logits to integer frequency tables (`probs.py`); wire up end-to-end text compression (`llmcodec.py`). ✅
4. Container format (`container.py`), zstd fallback (`fallback.py`), chunking, routing (`api.py`) and the real CLI. ✅
5. Benchmarks (`bench.py`, `llmzip bench`), weights-hash cache, LLM skip heuristic, per-token profile (`scripts/profile_step.py`). ✅
6. `inspect` heatmap (`inspect_cmd.py`), LLM token budget, wrap-up. ✅ (tagged v0.1.0)
7. Textual terminal app (`llmzip tui`). ✅
8. Restored copies with an identity check, TUI polish, Finder Quick Actions. ✅

All phases are done.

## Commands

```
uv run pytest
uv run python scripts/smoke_model.py
uv run llmzip info [--fingerprint]
uv run python scripts/demo_coder.py FILE...
uv run python scripts/demo_llm.py bench/samples/*
uv run python scripts/demo_cli.py            # CLI end to end: samples, random bytes, a PNG
uv run llmzip bench bench/samples/*          # benchmark table -> bench/results/
uv run python scripts/profile_step.py        # per-token time breakdown
uv run llmzip compress FILE && uv run llmzip decompress FILE.lz -o OUT
```

## Future ideas (collected from the phase reports)

- **Parallel block coding:** blocks are independent, so N processes can code N blocks at once with no format change. This is the biggest practical speed-up on a 10-core machine.
- **Leaner GPT-2 forward pass:** the HF forward is 93.5% of per-token time. A direct-module forward with a preallocated KV cache could be much faster, but it **must be bit-identical**. Verify with the fresh-process logits digest and a byte comparison of `.lz` files.
- **Sliding-window context as format v2:** recover the block-restart loss (+5.2% at 512-token blocks vs 1023). This needs a new container version and a matching decoder, and must stay incremental (HARD RULE 3).
- **A personal or domain fine-tuned model:** fine-tuning on your own text or code would lower bits/char a lot. It changes the fingerprint, so files are tied to that model.
- **Cross-machine portability:** bit-exactness across CPUs and OSes is untested. It could come from integer or fixed-point inference, or from pinning kernels, with the CPU type recorded in the fingerprint.
- **Smaller wins:**
  - torch float64 softmax and a fused cumsum in probs (up to about 6%; any change to freqs requires a `PROBS_SCHEME` bump);
  - a smarter skip estimate (sample several blocks);
  - GPT-2's tendency to expect another `\n` after a newline makes indentation in code expensive (see the phase 6 report).
