# Phase 6 report: inspect heatmap, LLM budget, wrap-up

## 1. Summary
`llmzip inspect` shows the per-token surprise exactly as compression pays it. It prints a colour heatmap that keeps the file's layout, then a summary, the top-N most surprising tokens with the model's top-3 guesses, the longest "autopilot" stretches, and optional JSON. Its predicted payload matches the real compressed payload exactly on all three demo files.

Compress gained a `--max-llm-tokens` budget (default 25000). New bench results are now gitignored, and the README and CLAUDE.md were finalised. Compression output is byte-identical to Phase 5, the fingerprint is unchanged (`2bb9a6826f48e26a`), and all 324 tests pass.

## 2. Environment
Unchanged: macOS 15.7.1, Apple M4 Pro, uv 0.13.1, Python 3.12.15, torch 2.14.1, transformers 5.20.1, numpy 2.5.4, zstandard 0.25.0, rich 15.0.0, xz 5.8.4, zstd 1.5.7, Apple gzip 457.140.3, git 2.50.1. No new dependencies.

## 3. Files created/changed
| File | Purpose |
|---|---|
| `src/llmzip/inspect_cmd.py` | Replaces the stub. `analyze()` (per-token ideal and quantized bits, top-3, blocks, in-pass coding), `token_segments()` (GPT-2 byte↔unicode table), top-N, autopilot runs, summary, JSON, rich rendering (heatmap with a 6-bucket fixed scale and legend, or a plain listing) |
| `src/llmzip/cli.py` | `inspect` subcommand; `compress --max-llm-tokens`; budget message in the summary line; `show_progress` tolerates commands without `--quiet` |
| `src/llmzip/api.py` | `max_llm_tokens` budget (checked after tokenizing, before any model work) and `CompressInfo.llm_over_budget` |
| `src/llmzip/bench.py` | The replayed auto choice respects the budget |
| `src/llmzip/config.py` | `LLM_TOKEN_BUDGET = 25000` |
| `tests/test_inspect.py` (new) | 12 tests (see section 7) |
| `.gitignore` | `bench/results/*` (the two committed result sets stay tracked; publish new ones with `git add -f`) |
| `README.md` | Quickstart at the top, Inspect section with an example, "How to read the heatmap", budget in Limitations |
| `CLAUDE.md` | All 6 phases marked done; inspect and budget conventions; "Future ideas" |
| `reports/phase-5.md` | Commit-hash edit carried over from Phase 5 |
| `reports/phase-6.md` | This report |

## 4. Key decisions
- **Exactly what compression sees:** `analyze` uses `tokenize_checked`, splits the ids into the same blocks, restarts each block with `model.start()`, and steps with `next_logits`. It reuses `probs.softmax64` for ideal bits and `probs.logits_to_freqs` for quantized bits; the probs logic isn't duplicated.
- **Actual payload without a second model pass:** each block is arithmetic-coded inside the same loop with the real `Encoder`, `cumulative` and `symbol_interval`. That doubles nothing.
  - To make sure this really is the compression output, two tests assert that inspect's block bytes **equal `compress_text`'s blocks, byte for byte**: for prose.txt with the default block size, and with `block_size=50` (6 blocks).
  - None of the compression modules changed.
- **Token text:** transformers 5 doesn't expose GPT-2's `byte_decoder`, so I rebuilt the standard `bytes_to_unicode` table (from the original GPT-2 encoder).
  - Each token's raw bytes are accumulated, and a character split across tokens is shown on the token that completes it.
  - `analyze` checks that the segments concatenate to the exact input, and a test with emoji, CJK, a combining accent, CRLF and tabs at `block_size=3` covers it.
- **Heatmap:**
  - Six fixed buckets of quantized bits/token: <0.5, 0.5–1.5, 1.5–3, 3–6, 6–12 and ≥12, running from dark blue through pale yellow to dark red. Readable foreground colours are chosen per bucket.
  - It is printed with `soft_wrap=True`, so rich never inserts line breaks. I verified this: with ANSI codes stripped, the heatmap output is exactly the original file for all three demo files.
  - Whitespace is shown only by its background colour. A one-line legend follows.
- **Plain mode** (`--no-color`, or stdout not a terminal): a `'token'[bits]` listing of the first 200 tokens, then the summary and the tables, without colour.
- **Summary:**
  - tokens, characters, blocks, quantized and ideal bits, bits/char;
  - predicted payload, the sum over blocks of ceil(quantized bits / 8), next to the actual payload;
  - predictability: the percentage of tokens costing under 1 bit.
- **Top-N:** sorted by quantized bits, descending, with ties kept in token order. It shows `repr` of the token, bits, probability, a context snippet with the token in `[...]`, and the top-3 predictions (from float64 softmax, sorted).
- **Autopilot stretches:** the 5 longest runs of consecutive tokens each under 0.5 quantized bits.
- **Truncation:** `--max-tokens` (default 5000, 0 = all) analyzes the first N tokens, and the output says "NOTE: output truncated to the first N of M tokens".
- **Errors:** non-UTF-8, empty or non-round-trippable input exits 1 with `llmzip: error: …`. The UTF-8 check runs before the model loads.
- **LLM budget:** in auto mode the token count is compared with the budget right after tokenizing, so no LLM block runs on over-budget text. Forced `--mode llm` ignores it, and 0 means unlimited.
  - The summary line says: `LLM skipped: N tokens > budget M (use --max-llm-tokens 0 for unlimited)`.
  - The default of 25000 tokens is about 3–4 minutes each way.

## 5. Commands run and results
| Command | Result | Key output |
|---|---|---|
| Phase 5 baseline (`compress --mode llm` on `bench/samples/*.txt` before any change) | pass | 2,196 / 199 / 526 B |
| `uv run llmzip inspect bench/samples/prose.txt --top 5` (plain, first try) | pass | predicted 157 B = actual 157 B |
| `uv run pytest tests/test_inspect.py` | pass | 11 passed in 18.11s (the 12th, the markup regression test, was added after the fix) |
| `inspect` with colour (`COLORTERM=truecolor TTY_COMPATIBLE=1 FORCE_COLOR=1`) on prose.txt, sample.py, sample.json → `$TMPDIR/inspect_*.ansi` | pass | 24-bit backgrounds; with ANSI stripped, the text equals the original file for all 3 |
| `inspect --no-color` on the same 3 files (prose.txt also with `--json $TMPDIR/inspect_prose.json`) | pass | see section 6; the JSON has 284 tokens, summary 157/157 B |
| Cross-check of a surprising code token with a plain batched HF forward pass | pass | p(`import`) = 8.3e-12, p(`\n`) = 0.9999: a model property, not an llmzip artifact |
| `uv run pytest` (after the markup fix) | pass | 324 passed in 168.57s |
| Byte identity vs Phase 5 (`*.txt`, `--mode llm` and auto) | pass | 6/6 identical |

## 6. Smoke test results

### Inspect demos (plain-text output, `--no-color`)

**bench/samples/prose.txt**
```
tokens 284  characters 1301  blocks 1  bits 1255.8 (ideal 1254.9)  bits/char 0.965
predicted LLM payload 157 B  actual payload 157 B  predictability 18.7% of tokens < 1 bit
```
| # | token | bits | prob | context | model's top 3 |
|---|---|---|---|---|---|
| 89 | `' no'` | 16.8 | 8.6e-06 | `ttle further every spring[ no] matter how oft` | '.' 0.67, ',' 0.07, ' when' 0.06 |
| 49 | `' strings'` | 16.3 | 1.2e-05 | `and beans that climbed up[ strings] she tied to th` | ' the' 0.35, ' to' 0.11, ' into' 0.05 |
| 173 | `' enormous'` | 15.0 | 3e-05 | `d just laugh, hold up the[ enormous] vegetable, and` | ' basket' 0.29, ' carrot' 0.02, ' little' 0.02 |
| 254 | `' reason'` | 13.9 | 6.6e-05 | `eally missed was having a[ reason] to go outside` | ' garden' 0.35, ' good' 0.07, ' little' 0.02 |
| 77 | `' stubborn'` | 13.6 | 8.3e-05 | `l, thyme, rosemary, and a[ stubborn] clump of mint` | ' few' 0.17, ' little' 0.15, ' handful' 0.03 |
| 144 | `' z'` | 13.5 | 8.4e-05 | ` still too small and left[ z]ucchini on the` | ' them' 0.68, ' the' 0.07, ' a' 0.04 |
| 65 | `' herbs'` | 13.4 | 9.6e-05 | `ddle there was a patch of[ herbs]: basil, thyme,` | ' grass' 0.11, ' land' 0.02, ' green' 0.02 |
| 45 | `' beans'` | 12.9 | 0.00013 | `ht the afternoon sun, and[ beans] that climbed u` | ' she' 0.08, ' kept' 0.05, ' then' 0.05 |
| 47 | `' climbed'` | 12.8 | 0.00014 | `rnoon sun, and beans that[ climbed] up strings she` | ' were' 0.13, ' she' 0.11, ' grew' 0.06 |
| 161 | `' sc'` | 11.7 | 0.0003 | ` baseball bats. She never[ sc]olded me. She w` | ' told' 0.07, ' let' 0.05, ' gave' 0.04 |

**bench/samples/sample.py**
```
tokens 309  characters 761  blocks 1  bits 1318.5 (ideal 1335.6)  bits/char 1.733
predicted LLM payload 165 B  actual payload 165 B  predictability 36.6% of tokens < 1 bit
```
| # | token | bits | prob | context | model's top 3 |
|---|---|---|---|---|---|
| 19 | `'import'` | 24.0 | 8.3e-12 | `on sample."""\n\nimport re\n[import] sys\nfrom colle` | '\n' 1.00, '.' 0.00, ',' 0.00 |
| 22 | `'from'` | 24.0 | 2.1e-09 | `""\n\nimport re\nimport sys\n[from] collections im` | '\n' 1.00, '.' 0.00, ',' 0.00 |
| 59 | `' '` | 22.0 | 2.2e-07 | `ignoring punctuation."""\n[ ]   return re.fi` | '\n' 1.00, '.' 0.00, '\n\n' 0.00 |
| 42 | `' '` | 21.4 | 3.5e-07 | `text: str) -> list[str]:\n[ ]   """Split tex` | '\n' 1.00, '.' 0.00, ',' 0.00 |
| 112 | `' counts'` | 21.4 | 3.2e-07 | `tuple[str, int]]:\n   [ counts] = Counter(toke` | ' """' 0.82, ' "' 0.04, '\n' 0.02 |
| 122 | `' '` | 20.3 | 7.5e-07 | ` Counter(tokenize(text))\n[ ]   return count` | '\n' 1.00, '\n\n' 0.00, '.' 0.00 |
| 150 | `' '` | 20.1 | 8.5e-07 | `argv: list[str]) -> int:\n[ ]   if len(argv` | '\n' 1.00, '\n\n' 0.00, '.' 0.00 |
| 193 | `' '` | 19.5 | 1.3e-06 | ` FILE", file=sys.stderr)\n[ ]       return` | '\n' 1.00, '\n\n' 0.00, '.' 0.00 |
| 163 | `' '` | 19.2 | 1.6e-06 | `   if len(argv) != 2:\n[ ]       print` | '\n' 1.00, '.' 0.00, '\n\n' 0.00 |
| 294 | `' '` | 19.2 | 1.7e-06 | ` __name__ == "__main__":\n[ ]   sys.exit(mai` | '\n' 1.00, '\n\n' 0.00, '.' 0.00 |

These rows come from the re-run after the markup fix (section 8). The first run had dropped `[import]`, `[from]` and the `[str]` in type hints.

**bench/samples/sample.json**
```
tokens 358  characters 775  blocks 1  bits 869.4 (ideal 868.2)  bits/char 1.122
predicted LLM payload 109 B  actual payload 109 B  predictability 61.5% of tokens < 1 bit
```
| # | token | bits | prob | context | model's top 3 |
|---|---|---|---|---|---|
| 2 | `' '` | 20.0 | 9.1e-07 | `{\n[ ] "library": {` | '\n' 1.00, '.' 0.00, '"' 0.00 |
| 348 | `' '` | 18.5 | 2.7e-06 | `2019, "available": true}\n[ ]   ]\n  }` | '\n' 1.00, ']' 0.00, '<\|endoftext\|>' 0.00 |
| 353 | `' '` | 17.0 | 7.8e-06 | `"available": true}\n    ]\n[ ] }\n}` | '\n' 1.00, '}' 0.00, ']' 0.00 |
| 213 | `' null'` | 16.6 | 9.9e-06 | `",\n      "sunday":[ null]\n    },` | ' "' 1.00, ' {' 0.00, ' ""' 0.00 |
| 4 | `'library'` | 15.8 | 1.7e-05 | `{\n  "[library]": {\n    "` | ' name' 0.08, ' type' 0.02, ' id' 0.02 |
| 8 | `' '` | 15.7 | 1.8e-05 | `{\n  "library": {\n[ ]   "name": "R` | '\n' 1.00, ' "' 0.00, '.' 0.00 |
| 74 | `' '` | 15.4 | 2.3e-05 | `  "postal_code": "10217"\n[ ]   },` | '\n' 1.00, ' "' 0.00, ',' 0.00 |
| 244 | `' Harbor'` | 15.1 | 2.9e-05 | `: 1, "title": "The Silent[ Harbor]", "author": "M` | ' Book' 0.05, ' Spring' 0.03, ' Majority' 0.02 |
| 215 | `' '` | 14.9 | 3.3e-05 | `      "sunday": null\n[ ]   },` | '\n' 1.00, '}' 0.00, '};' 0.00 |
| 22 | `' '` | 14.8 | 3.5e-05 | `verside Public Library",\n[ ]   "address": {` | '\n' 1.00, ' "' 0.00, '.' 0.00 |

Longest autopilot stretches in sample.json: 10 tokens (0.8 bits) of `:00",\n      "` three times (the opening-hours lines), 9 tokens of ` {"id": 3, "title": "`, and 7 tokens of `nesday": "09:00-`.

**Finding: GPT-2 and line starts in code.** In code and JSON, the most expensive tokens are mostly the *first token of a line*: indentation `' '` or `import`/`from`. After a `\n`, GPT-2 predicts yet another `\n` with p ≈ 1.00, so the real line start costs 15–24 bits; 24.0 is the floor, frequency 1 out of 2^24.
- A plain batched HF forward pass confirms p(`import` | `…import re\n`) = 8.3e-12. This is a property of GPT-2 small, not of llmzip.
- The 10 worst tokens in sample.py alone cost about 210 bits, about 16% of the file.
- This is a concrete target for future work; see section 12.

### JSON output
`$TMPDIR/inspect_prose.json` (42,770 B) holds 284 per-token records `{id, text, ideal_bits, quantized_bits, block}`, the summary (predicted 157 = actual 157 B, ideal 1254.9 bits), the top-10 entries with their top-3 and the autopilot runs. The first record is `{"id": 3666, "text": "My", "ideal_bits": 8.871, "quantized_bits": 8.875, "block": 0}`.

### Byte identity vs Phase 5
I made the baseline `.lz` files with the Phase 5 code (HEAD `1aec5d5`, before any Phase 6 edit), then compressed again after all changes:

| file | `--mode llm` | auto mode |
|---|---|---|
| blocktest.txt (2,196 B) | ✅ identical | ✅ identical |
| prose.txt (199 B) | ✅ identical | ✅ identical |
| prose_long.txt (526 B) | ✅ identical | ✅ identical |

`test_loads_without_network` confirms the fingerprint is still `2bb9a6826f48e26a`.

## 7. Test results
`324 passed in 168.57s`: Phase 5's 312 plus 12 new tests in `tests/test_inspect.py`.
- **Payload:** the predicted payload equals the per-block ceiling sum of quantized bits and is within 4 B of actual (it was 0 B off for prose.txt). Inspect's block bytes equal `compress_text`'s.
- **Multi-block (`block_size=50`):** the block bytes equal compression's, and the block indices are correct.
- **Segments:** they reproduce the text exactly with emoji, CJK, a combining accent, CRLF and tabs at `block_size=3`.
- **Ideal bits:** the Phase 1 smoke paragraph gives 1151.8 ± 0.1, matching Phase 1.
- **Rendering:** a regression test checks that bracketed context such as `[import]` survives (rich markup).
- **Top-10:** sorted by bits, descending, and the first entry is the maximum. Each top-3 has valid token ids and strings, with probabilities sorted and summing to at most 1.
- **Autopilot runs:** sorted by length, every token in them is under 0.5 bits.
- **`--max-tokens`:** truncation is flagged and the count is right.
- **CLI:** `--no-color --json --max-tokens 50 --top 5` gives valid JSON with 50 tokens, prints the truncation note, and has no ANSI codes. Non-UTF-8 input exits 1 with a clean message.
- **Budget:** a budget of 10 in auto mode gives ZSTD or STORED and round-trips; forced `llm` ignores the budget; a budget of 0 is unlimited and matches forced output byte for byte. The CLI budget message is checked.

## 8. Deviations from the plan
- **Actual payload:** computed by coding inside inspect's own pass instead of calling `compress_text` again, which avoids a second, equally slow model pass. Equality with `compress_text` is asserted in two tests.
- **Colour capture:** the sandbox has no pty, so I forced rich's terminal mode (`TTY_COMPATIBLE=1 FORCE_COLOR=1 COLORTERM=truecolor`) and saved the ANSI output to `$TMPDIR/inspect_{prose.txt,sample.py,sample.json}.ansi`. I verified the layout programmatically. No human-viewed screenshot was taken; `cat` those files in a truecolor terminal to see them.
- **Heatmap wrap fix:** the first colour capture showed that rich inserted line breaks at the console width. I switched to `soft_wrap=True` and re-verified.
- **Bug found and fixed during the demo:** rich parsed bracketed context such as `[import]`, `[library]` or `list[str]` as markup tags and silently dropped them from the tables. Prose rows like `[ no]` survived only because a leading space isn't valid markup.
  - Table cells are now `rich.text.Text` objects, so they are never parsed as markup.
  - A regression test (`test_render_keeps_bracketed_context`) covers it.
  - The sample.py and sample.json tables above come from the fixed re-run.
- **Extras:**
  - `--max-tokens 0` means all tokens;
  - empty input exits 1;
  - autopilot runs also go into the JSON;
  - bench respects the budget.

## 9. Problems and open questions
- **No unresolved failures.**
- **Final report hash:** as in every phase, the hash in section 11 was written after the commit, so it's an uncommitted one-line edit. There is no Phase 7 to carry it, so it needs a decision from you: a small follow-up commit, or leave it uncommitted.
- **Whitespace in context cells:** whitespace-only tokens appear in the context column as `[ ]`, which is easy to misread. Showing them as `␠` would be a small UX improvement; it's left as is.
- **Heatmap colours** are tuned for dark and light terminals with truecolor. 16-colour terminals get rich's nearest-colour mapping, which merges some buckets.
- **Inspect speed:** inspect takes about as long as compression itself (5000 tokens is about 40–50 s).

## 10. Notes for later phases
There are no later phases. CLAUDE.md now carries the "Future ideas" list, summarised in section 12.

## 11. Git
- Commit: `<HASH>`, tagged `v0.1.0`
- Message: `Phase 6: inspect heatmap, LLM budget, wrap-up`
- The hash was written into this file after the commit, so this line is an uncommitted edit (see section 9).

## 12. Project wrap-up

**What llmzip does.** It compresses files losslessly by letting GPT-2 (124M, CPU, deterministic) predict each next token. An integer arithmetic coder then spends about −log2 p bits on the token that actually occurs. Decompression replays the identical model to recover every token. Around that core:
- **Container:** a versioned `.lz` file with CRC32 and a model fingerprint.
- **Routing:** "never worse" routing to zstd or stored, plus a skip heuristic and a token budget.
- **Long text:** 1023-token blocks for text of any length.
- **CLI:** clean errors, atomic writes and a progress bar.
- **Tools:** `bench` and the `inspect` heatmap.

**Final test count: 324 passing** (217 coder, 28 codec, 36 container, 22 CLI/API, 9 model, 12 inspect).

**Headline benchmark numbers** (Apple M4 Pro, 1 CPU thread; sizes include the about 42 B container header):

| sample | llmzip | best classic (gzip/xz/zstd) | gain |
|---|---|---|---|
| English prose, 4.1 KB | **526 B (1.01 bits/byte)** | 1,872 B (zstd -19) | 3.6× smaller |
| English prose, 1.3 KB | **199 B (1.22 bits/byte)** | 697 B (gzip -9) | 3.5× smaller |
| Markdown, JSON, Python snippets | **150–207 B (1.5–2.2 bits/byte)** | 365–444 B | 2.1–2.8× smaller |
| Technical reports, 32.9 KB | **7,419 B (1.80 bits/byte)** | 12,104 B (xz -9) | 1.6× smaller |
| Repetitive 200 KB log | 113 B via zstd (LLM skipped in 9.9 s) | — | — |

Throughput is about 100–160 tokens/s each way. Byte-exact round trips held across separate processes, through every phase.

**Known limitations:**
- **Slow:** the GPT-2 forward pass is 93.5% of the time. 1 MB of text takes about 30 minutes, so auto mode caps the LLM at 25k tokens by default.
- **Decompression needs the identical stack:** the same weights, torch, transformers and numpy, and in practice the same machine type. The fingerprint enforces the software side; cross-CPU bit-exactness is untested.
- **UTF-8 text only:** only valid UTF-8 text benefits.
- **Block restarts:** each 1023-token block starts cold, costing a few percent on long files.
- **GPT-2 small** is weak on code and structured text; line-start tokens are especially costly.

**Top 3 next steps:**
1. **Parallel block coding:** blocks are already independent, so N worker processes give close to N× faster compression and decompression with no format change. This addresses the biggest limitation.
2. **A leaner, bit-identical GPT-2 forward pass:** direct module calls and a preallocated KV cache would cut the per-token overhead that dominates the profile. It must be verified with the fresh-process logits digest and `.lz` byte identity, as in Phases 5 and 6.
3. **Better model and context (format v2):** sliding-window context to recover the block-restart loss, and a stronger or fine-tuned model to fix weaknesses such as GPT-2 small's line-start problem in code. Either one gets a new fingerprint and format version.
