# llmzip

Lossless text compression with a local GPT-2 model and arithmetic coding. Each token costs about `-log2 p(token)` bits under the model's prediction, so predictable text compresses far better than with gzip. This runs on CPU only and is deterministic, so compression and decompression produce identical probabilities.

Status: phase 3 of 6. In-memory GPT-2 compression of texts of up to 1023 tokens works (`llmzip.llmcodec`); about 0.9–1.7 bits/char on the samples, 2.7–4.4× smaller than gzip -9. The CLI `compress`, `decompress`, `inspect` and `bench` commands are not implemented yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The first run downloads GPT-2 (~500 MB) into the Hugging Face cache.

```
uv sync
uv run llmzip info               # configuration and versions
uv run llmzip info --fingerprint # also load the model and print its fingerprint
uv run python scripts/smoke_model.py  # determinism, speed, ideal compressed size
uv run python scripts/demo_coder.py README.md  # coder demo with an adaptive byte model
uv run python scripts/demo_llm.py bench/samples/*  # GPT-2 compression, fresh-process decompress
uv run pytest
```
