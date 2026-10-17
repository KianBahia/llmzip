# llmzip

Lossless file compression with a local GPT-2 model and arithmetic coding. Each token costs about `-log2 p(token)` bits under the model's prediction, so predictable text compresses far better than with gzip. It runs on CPU only and is deterministic, so compression and decompression produce identical probabilities.

Status: phase 4 of 6. `compress` and `decompress` work on any file.
- **Text** is coded with GPT-2 in blocks of up to 1023 tokens.
- **Everything else** goes to zstd or is stored as-is, whichever is smallest.
- **Integrity:** every `.lz` file records a CRC32 and the model fingerprint.
- **Not yet implemented:** `inspect` and `bench`.

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
- `-o OUT`;
- `--force` to overwrite;
- `-q` for quiet.

Exit codes are 0 ok, 1 error, 2 usage error. Set `LLMZIP_DEBUG=1` for tracebacks.

Speed is about 100 tokens/s in each direction (about 4 KB of text per 10 s).

## Development

```
uv run pytest
uv run python scripts/demo_cli.py             # CLI end to end on samples + binaries
uv run python scripts/demo_llm.py bench/samples/*
```
