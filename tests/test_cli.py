import os

import pytest

from llmzip import api
from llmzip import container as ct
from llmzip.errors import ContainerError, FingerprintMismatch, LlmzipError
from llmzip.llmcodec import compress_text, decompress_text
from llmzip.model import NextTokenModel

from helpers import PARAGRAPH, SAMPLES, run_cli, text_with_exactly


@pytest.fixture(scope="module")
def model():
    return NextTokenModel()


def roundtrip_bytes(data: bytes, model, **kw) -> api.CompressInfo:
    blob, info = api.compress_bytes(data, model=model, **kw)
    restored, c = api.decompress_bytes(blob, model=model)
    assert restored == data
    assert c.mode == info.mode
    assert len(blob) == info.compressed_size
    return info


# --- routing ----------------------------------------------------------------------


def test_empty_is_stored(model):
    info = roundtrip_bytes(b"", model)
    assert info.mode == ct.STORED and info.compressed_size == ct.HEADER_SIZE


def test_random_bytes_not_llm(model):
    data = os.urandom(4096)
    info = roundtrip_bytes(data, model)
    assert info.mode in (ct.ZSTD, ct.STORED)
    assert info.compressed_size <= len(data) + ct.HEADER_SIZE


def test_invalid_utf8_is_zstd(model):
    data = PARAGRAPH.encode() + b"\xff" + PARAGRAPH.encode()
    info = roundtrip_bytes(data, model)
    assert info.mode == ct.ZSTD
    assert "LLM" not in info.candidates


def test_tiny_text_picks_smallest(model):
    info = roundtrip_bytes(b"hello", model)
    assert info.compressed_size == min(info.candidates.values())
    assert info.mode == ct.STORED


def test_prose_is_llm(model):
    info = roundtrip_bytes(PARAGRAPH.encode(), model)
    assert info.mode == ct.LLM
    assert info.compressed_size < min(v for k, v in info.candidates.items() if k != "LLM")


def test_utf8_bom_roundtrips_byte_identical(model):
    data = b"\xef\xbb\xbf" + PARAGRAPH.encode()
    info = roundtrip_bytes(data, model)
    assert info.mode == ct.LLM  # strict utf-8 keeps the BOM as U+FEFF text


def test_not_round_trippable_text_goes_zstd(model, monkeypatch):
    import llmzip.api as api_mod
    from llmzip.llmcodec import NotRoundTrippable

    def refuse(text, m):
        raise NotRoundTrippable("forced for test")

    monkeypatch.setattr(api_mod, "tokenize_checked", refuse)
    info = roundtrip_bytes(PARAGRAPH.encode(), model)
    assert info.mode == ct.ZSTD
    with pytest.raises(NotRoundTrippable):
        api.compress_bytes(PARAGRAPH.encode(), mode="llm", model=model)


def test_forced_modes(model):
    data = PARAGRAPH.encode()
    assert roundtrip_bytes(data, model, mode="store").mode == ct.STORED
    assert roundtrip_bytes(data, model, mode="zstd").mode == ct.ZSTD
    assert roundtrip_bytes(b"hi", model, mode="llm").mode == ct.LLM  # forced even if larger
    with pytest.raises(LlmzipError, match="UTF-8"):
        api.compress_bytes(b"\xff\xfe", mode="llm", model=model)


# --- corruption -------------------------------------------------------------------


def test_flipped_llm_payload_byte_detected(model):
    blob, info = api.compress_bytes(PARAGRAPH.encode(), mode="llm", model=model)
    corrupted = bytearray(blob)
    corrupted[-20] ^= 0x10  # inside the block payload
    with pytest.raises(ContainerError):
        api.decompress_bytes(bytes(corrupted), model=model)


class NoDecodeModel:
    """Spy: reports a fingerprint but fails if any model computation is attempted."""

    def __init__(self, real):
        self._fp = real.fingerprint_bytes()

    def fingerprint_bytes(self):
        return self._fp

    def start(self):
        raise AssertionError("model ran before the fingerprint check")

    next_logits = start


def test_fingerprint_mismatch_before_decoding(model):
    blob, _ = api.compress_bytes(PARAGRAPH.encode(), mode="llm", model=model)
    corrupted = bytearray(blob)
    corrupted[ct.HEADER_SIZE] ^= 0xFF  # first fingerprint byte
    with pytest.raises(FingerprintMismatch):
        api.decompress_bytes(bytes(corrupted), model=NoDecodeModel(model))


# --- chunking ---------------------------------------------------------------------


def test_chunking_partial_last_block(model):
    text = (SAMPLES / "prose.txt").read_bytes().decode("utf-8")
    blocks, n = compress_text(text, model, block_size=50)
    assert n % 50 != 0 and len(blocks) == -(-n // 50) > 1
    assert decompress_text(blocks, n, model, block_size=50) == text


def test_chunking_exact_multiple(model):
    text = text_with_exactly(300, model)
    blocks, n = compress_text(text, model, block_size=50)
    assert n == 300 and len(blocks) == 6
    assert decompress_text(blocks, n, model, block_size=50) == text


def test_chunking_splits_multibyte_characters(model):
    text = "日本語のテキスト 🍕🔥 café 中文文本 👩‍👩‍👧 end"
    for block_size in (1, 2, 3, 7):
        info = roundtrip_bytes(text.encode(), model, mode="llm", block_size=block_size)
        assert info.mode == ct.LLM


def test_chunking_via_api_container(model):
    data = (SAMPLES / "prose.txt").read_bytes()
    blob, _ = api.compress_bytes(data, mode="llm", block_size=50, model=model)
    c = ct.unpack(blob)
    assert c.body.block_size == 50 and len(c.body.blocks) == -(-c.body.n_tokens // 50)
    assert api.decompress_bytes(blob, model=model)[0] == data


# --- CLI in separate processes ----------------------------------------------------


@pytest.mark.slow
def test_cli_two_block_roundtrip_separate_processes(tmp_path):
    src = tmp_path / "long.txt"
    src.write_bytes((SAMPLES / "prose_long.txt").read_bytes() + (SAMPLES / "prose.txt").read_bytes())
    run_cli("compress", src, "--quiet")
    lz = tmp_path / "long.txt.lz"
    c = ct.unpack(lz.read_bytes())
    assert c.mode == ct.LLM and len(c.body.blocks) == 2 and 1023 < c.body.n_tokens <= 2046
    out = tmp_path / "restored.txt"
    run_cli("decompress", lz, "-o", out, "--quiet")
    assert out.read_bytes() == src.read_bytes()


@pytest.mark.slow
def test_cli_default_names_and_verify(tmp_path):
    src = tmp_path / "note.md"
    src.write_bytes((SAMPLES / "sample.md").read_bytes())
    r = run_cli("compress", src, "--verify")
    assert "mode=LLM" in r.stdout and "verified" in r.stdout
    src.rename(tmp_path / "original.md")
    run_cli("decompress", tmp_path / "note.md.lz", "--quiet")
    assert (tmp_path / "note.md").read_bytes() == (tmp_path / "original.md").read_bytes()


def test_cli_refuses_overwrite_without_force(tmp_path):
    src = tmp_path / "a.txt"
    src.write_bytes(b"x")
    (tmp_path / "a.txt.lz").write_bytes(b"existing")
    r = run_cli("compress", src, check=False)
    assert r.returncode == 1 and "--force" in r.stderr
    assert (tmp_path / "a.txt.lz").read_bytes() == b"existing"
    assert run_cli("compress", src, "--force", "--quiet").returncode == 0
    assert src.read_bytes() == b"x"  # input never touched


def test_cli_decompress_non_lz_name_needs_output(tmp_path):
    f = tmp_path / "foo.bin"
    f.write_bytes(b"whatever")
    r = run_cli("decompress", f, check=False)
    assert r.returncode == 2 and "-o" in r.stderr


def test_cli_corrupt_file_clean_error_no_output(tmp_path):
    src = tmp_path / "z.txt"
    src.write_bytes(PARAGRAPH.encode() * 3)
    run_cli("compress", src, "--mode", "zstd", "--quiet")
    lz = tmp_path / "z.txt.lz"
    blob = bytearray(lz.read_bytes())
    blob[-3] ^= 0x01
    lz.write_bytes(bytes(blob))
    out = tmp_path / "restored.txt"
    r = run_cli("decompress", lz, "-o", out, check=False)
    assert r.returncode == 1
    assert r.stderr.startswith("llmzip: error:") and "Traceback" not in r.stderr
    assert not out.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_cli_bad_magic_exit_1(tmp_path):
    f = tmp_path / "bad.lz"
    f.write_bytes(b"not an llmzip file")
    r = run_cli("decompress", f, check=False)
    assert r.returncode == 1 and "bad magic" in r.stderr
