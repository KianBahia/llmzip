# llmzip

Lossless text compression with a local GPT-2 model and arithmetic coding. Each token costs about `-log2 p(token)` bits under the model's prediction, so predictable text compresses far better than with gzip. This runs on CPU only and is deterministic, so compression and decompression produce identical probabilities.

Status: phase 1 of 6. The model wrapper and the smoke test work; `compress`, `decompress`, `inspect` and `bench` are not implemented yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The first run downloads GPT-2 (~500 MB) into the Hugging Face cache.

```
uv sync
uv run llmzip info               # configuration and versions
uv run llmzip info --fingerprint # also load the model and print its fingerprint
uv run python scripts/smoke_model.py  # determinism, speed, ideal compressed size
uv run pytest
```
