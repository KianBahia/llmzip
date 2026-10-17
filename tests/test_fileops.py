from pathlib import Path

import pytest

from llmzip.errors import UsageError
from llmzip.fileops import identity_check, restored_output, unique_path


@pytest.mark.parametrize("lz, restored", [
    ("report.md.lz", "report.restored.md"),
    ("notes.lz", "notes.restored"),
    ("a.tar.gz.lz", "a.tar.restored.gz"),
    ("my notes.lz", "my notes.restored"),          # spaces, no inner extension
    ("my notes.txt.lz", "my notes.restored.txt"),
    (".bashrc.lz", ".bashrc.restored"),
    ("report-2.md.lz", "report-2.restored.md"),
])
def test_restored_output_naming(lz, restored, tmp_path):
    assert restored_output(tmp_path / lz) == tmp_path / restored


@pytest.mark.parametrize("name", ["report.md", "archive.lzma", ".lz"])
def test_restored_output_needs_lz(name, tmp_path):
    with pytest.raises(UsageError):
        restored_output(tmp_path / name)


def test_unique_path_last_extension(tmp_path):
    p = tmp_path / "report.restored.md"
    assert unique_path(p) == p
    p.write_text("x")
    assert unique_path(p) == tmp_path / "report.restored-2.md"
    (tmp_path / "report.restored-2.md").write_text("x")
    assert unique_path(p) == tmp_path / "report.restored-3.md"
    q = tmp_path / "notes.restored"
    q.write_text("x")
    assert unique_path(q) == tmp_path / "notes-2.restored"


@pytest.mark.parametrize("name, second, third", [
    ("report.md.lz", "report-2.md.lz", "report-3.md.lz"),
    ("a.tar.gz.lz", "a-2.tar.gz.lz", "a-3.tar.gz.lz"),
    ("notes.lz", "notes-2.lz", "notes-3.lz"),
    ("my notes.md.lz", "my notes-2.md.lz", "my notes-3.md.lz"),
])
def test_unique_path_base_for_lz_outputs(name, second, third, tmp_path):
    (tmp_path / name).write_text("x")
    assert unique_path(tmp_path / name, where="base") == tmp_path / second
    (tmp_path / second).write_text("x")
    assert unique_path(tmp_path / name, where="base") == tmp_path / third


def test_numbered_lz_keeps_inner_extension_for_restore(tmp_path):
    lz = tmp_path / "report.md.lz"
    lz.write_text("x")
    second = unique_path(lz, where="base")
    assert restored_output(second) == tmp_path / "report-2.restored.md"


def test_unique_path_rejects_bad_where(tmp_path):
    with pytest.raises(ValueError):
        unique_path(tmp_path / "x", where="middle")


def test_identity_check(tmp_path):
    lz = tmp_path / "report.md.lz"
    restored = tmp_path / "report.restored.md"
    restored.write_bytes(b"hello\n")
    assert identity_check(lz, restored) is None  # original not present

    (tmp_path / "report.md").write_bytes(b"hello\n")
    ident = identity_check(lz, restored)
    assert ident.identical and ident.original.name == "report.md"
    assert ident.original_sha256 == ident.restored_sha256 and ident.original_size == 6
    line = ident.line()
    assert "IDENTICAL ✓" in line and "report.md (6 B, " in line and ident.original_sha256[:12] in line

    (tmp_path / "report.md").write_bytes(b"hellO\n")
    ident = identity_check(lz, restored)
    assert not ident.identical and "DIFFERENT ✗" in ident.line() and " ≠ " in ident.line()

    # restoring over the original itself: nothing to compare against
    assert identity_check(lz, tmp_path / "report.md") is None


def test_identity_check_numbered_archive(tmp_path):
    (tmp_path / "report-2.md").write_text("same")
    (tmp_path / "report-2.restored.md").write_text("same")
    ident = identity_check(tmp_path / "report-2.md.lz", tmp_path / "report-2.restored.md")
    assert ident is not None and ident.identical and ident.original == tmp_path / "report-2.md"
