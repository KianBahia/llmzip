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
- Commit: `<HASH>`
- Message: `Phase 3: GPT-2 compression end to end`
- The hash was written into this file after the commit, so this line is an uncommitted edit that goes into the Phase 4 commit.
