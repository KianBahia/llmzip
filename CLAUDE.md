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
- **Offline:** `model.py` sets `HF_HUB_OFFLINE=1` when the pinned snapshot is already cached, and loads with `local_files_only=True` first. Import `llmzip.model` before `transformers`.
- Each phase writes a report to `reports/phase-N.md`.

## Phase roadmap

1. Setup, deterministic incremental model, smoke test. ✅
2. Integer-only arithmetic coder (`coder.py`) plus round-trip tests on synthetic distributions. ✅
3. Logits to integer frequency tables (`probs.py`); wire up end-to-end text compression.
4. Container format (`container.py`: mode, fingerprint, length, CRC32) and zstd fallback (`fallback.py`).
5. Benchmarks against gzip/xz/zstd (`bench`).
6. `inspect` heatmap (`inspect_cmd.py`) and polish.

## Commands

```
uv run pytest
uv run python scripts/smoke_model.py
uv run llmzip info [--fingerprint]
uv run python scripts/demo_coder.py FILE...
```
