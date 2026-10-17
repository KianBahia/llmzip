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

# Frequency tables (phase 3) sum to 2**FREQ_TOTAL_BITS. With a 50257-token vocab
# and a floor of 1 per token, the floor costs 50257 / 2**24 ~= 0.3% of the mass.
FREQ_TOTAL_BITS = 24

# Width of the arithmetic coder's low/high registers.
CODER_PRECISION_BITS = 32

# Coder invariant: after renormalization the coder's range (high - low + 1) is
# always > 2**(CODER_PRECISION_BITS - 2) (it never straddles the middle quarter
# without being shifted out). If total <= 2**(CODER_PRECISION_BITS - 2) <= range,
# then every symbol with freq >= 1 gets a sub-interval of width
# >= floor(range * freq / total) >= 1, so no symbol is ever unencodable.
assert FREQ_TOTAL_BITS <= CODER_PRECISION_BITS - 2, "FREQ_TOTAL_BITS too large for coder precision"
