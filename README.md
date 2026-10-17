# llmzip

Lossless file compression with a local GPT-2 model and arithmetic coding. Each token costs about `-log2 p(token)` bits under the model's prediction, so predictable text compresses far better than with gzip. It runs on CPU only and is deterministic, so compression and decompression produce identical probabilities.

Status: phase 5 of 6. `compress` and `decompress` work on any file.
- **Text** is coded with GPT-2 in blocks of up to 1023 tokens.
- **Everything else** goes to zstd or is stored as-is, whichever is smallest.
- **Integrity:** every `.lz` file records a CRC32 and the model fingerprint.
- **Not yet implemented:** `inspect`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The first run downloads GPT-2 (~500 MB) into the Hugging Face cache; later runs are fully offline.

```
uv sync
uv run llmzip info --fingerprint
```

## Usage

```
uv run llmzip compress notes.txt              # writes notes.txt.lz
uv run llmzip compress notes.txt --verify     # also decompress in-process and compare
uv run llmzip decompress notes.txt.lz -o restored.txt
```

Options:
- `--mode auto|llm|zstd|store` (default `auto`: smallest wins);
- `--no-skip` to always try the full LLM in auto mode;
- `-o OUT`;
- `--force` to overwrite;
- `-q` for quiet.

Exit codes are 0 ok, 1 error, 2 usage error. Set `LLMZIP_DEBUG=1` for tracebacks.

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

## Limitations

- **Slow:** about 100–150 tokens/s in each direction on one CPU core (about 4 KB of text per 10 s); 1 MB of text takes roughly half an hour.
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
