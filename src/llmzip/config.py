"""Single source of truth for settings that affect compressed output.

Anything here that changes the model's numbers (model, revision, threads,
attention implementation) is folded into the model fingerprint.
"""

# Model. MODEL_REVISION is the Hugging Face commit sha of the `gpt2` repo,
# fetched with huggingface_hub.model_info("gpt2").sha on 2026-10-17.
MODEL_NAME = "gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
ATTN_IMPLEMENTATION = "eager"

# GPT-2's <|endoftext|> token, fed first so the first real token has a context.
BOS_TOKEN_ID = 50256

# CPU threads for torch. Chosen from scripts/smoke_model.py measurements.
NUM_THREADS = 1

# Phase 3: frequency tables sum to 2**FREQ_TOTAL_BITS.
FREQ_TOTAL_BITS = 16
