"""Deterministic, incremental GPT-2 next-token logits.

Compressor and decompressor must both get their probabilities from
`NextTokenModel.start()` / `NextTokenModel.next_logits()` (HARD RULE 3): one token
per forward pass, reusing the KV cache. Never use a batched forward pass for coding.
"""

import hashlib
import sys

import torch
import transformers
from transformers import AutoTokenizer, GPT2LMHeadModel

from llmzip import config


def _apply_determinism(num_threads: int) -> None:
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)


class NextTokenModel:
    """GPT-2 on CPU, float32, eval mode, fed one token at a time.

    Sequence start: `start()` feeds BOS_TOKEN_ID (<|endoftext|>) and returns the
    logits for the first real token. Each `next_logits(t)` then feeds token t and
    returns the logits for the token after it.
    """

    def __init__(self, num_threads: int = config.NUM_THREADS):
        self.num_threads = num_threads
        _apply_determinism(num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME, revision=config.MODEL_REVISION
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            config.MODEL_NAME,
            revision=config.MODEL_REVISION,
            dtype=torch.float32,
            attn_implementation=config.ATTN_IMPLEMENTATION,
        ).to("cpu")
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size
        self.max_positions = self.model.config.n_positions
        self._weights_sha: str | None = None
        self.reset()

    def set_num_threads(self, num_threads: int) -> None:
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)

    def reset(self) -> None:
        self._cache = None
        self._pos = 0

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def next_logits(self, token_id: int) -> torch.Tensor:
        """Feed one token; return float32 logits (vocab_size,) for the next position."""
        if self._pos >= self.max_positions:
            raise ValueError(
                f"context window full ({self.max_positions} positions); "
                "long inputs are not handled yet"
            )
        input_ids = torch.tensor([[token_id]], dtype=torch.long)
        out = self.model(input_ids=input_ids, past_key_values=self._cache, use_cache=True)
        self._cache = out.past_key_values
        self._pos += 1
        return out.logits[0, -1].clone()

    def start(self) -> torch.Tensor:
        """Reset, feed BOS, and return the logits for the first real token."""
        self.reset()
        return self.next_logits(config.BOS_TOKEN_ID)

    def logits_sequence(self, ids: list[int]) -> list[torch.Tensor]:
        """Logits predicting each of `ids`, computed incrementally (len(ids) tensors)."""
        if not ids:
            return []
        out = [self.start()]
        for t in ids[:-1]:
            out.append(self.next_logits(t))
        return out

    def weights_sha256(self) -> str:
        if self._weights_sha is None:
            h = hashlib.sha256()
            for name, tensor in sorted(self.model.state_dict().items()):
                h.update(name.encode())
                h.update(tensor.contiguous().numpy().tobytes())
            self._weights_sha = h.hexdigest()
        return self._weights_sha

    def model_fingerprint(self) -> str:
        """16-hex-char id of everything that determines the logits."""
        parts = [
            config.MODEL_NAME,
            config.MODEL_REVISION,
            self.weights_sha256(),
            torch.__version__,
            transformers.__version__,
            f"threads={self.num_threads}",
            f"attn={config.ATTN_IMPLEMENTATION}",
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def logits_digest(model: NextTokenModel, text: str) -> str:
    """SHA-256 over the raw bytes of every incremental logits tensor for `text`."""
    h = hashlib.sha256()
    for logits in model.logits_sequence(model.encode(text)):
        h.update(logits.numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    # Fresh-process digest for cross-process determinism checks:
    #   python -m llmzip.model <num_threads> < text
    print(logits_digest(NextTokenModel(int(sys.argv[1])), sys.stdin.read()))
