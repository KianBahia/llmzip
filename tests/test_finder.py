"""Finder Quick Action scripts, run directly (not via Finder).

Notifications go to a recorder stub (LLMZIP_NOTIFY_CMD), the log to a tmp file
(LLMZIP_LOG_FILE), and install.sh only ever writes into a tmp --dest.
"""

import os
import plistlib
import shutil
import subprocess
from pathlib import Path

import pytest

from helpers import ROOT, SAMPLES

FINDER = ROOT / "finder"
UV = shutil.which("uv")
pytestmark = pytest.mark.skipif(UV is None, reason="uv not on PATH")


@pytest.fixture
def env(tmp_path):
    record = tmp_path / "notifications.txt"
    stub = tmp_path / "notify-stub.sh"
    stub.write_text(f'#!/bin/sh\nprintf "%s\\t%s\\n" "$1" "$2" >> "{record}"\n')
    stub.chmod(0o755)
    e = dict(os.environ, LLMZIP_NOTIFY_CMD=str(stub), LLMZIP_LOG_FILE=str(tmp_path / "finder.log"),
             LLMZIP_UV=UV, LLMZIP_REPO=str(ROOT))
    e.pop("LLMZIP_DRY_RUN", None)
    return e


def notifications(tmp_path) -> list[tuple[str, str]]:
    f = tmp_path / "notifications.txt"
    lines = f.read_text(encoding="utf-8").splitlines() if f.exists() else []
    out = []
    for line in lines:  # a message can span two lines (decompress: identity + summary)
        if "\t" in line:
            out.append(tuple(line.split("\t", 1)))
        elif out:
            out[-1] = (out[-1][0], out[-1][1] + "\n" + line)
    return out


def run_script(name, *files, env):
    return subprocess.run([str(FINDER / name), *map(str, files)], env=env,
                          capture_output=True, text=True, timeout=300)


def make_files(tmp_path, *names):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    out = []
    for n in names:
        p = work / n
        p.write_bytes((SAMPLES / "sample.md").read_bytes())
        out.append(p)
    return out


@pytest.mark.slow
def test_compress_then_decompress_identical(tmp_path, env):
    (src,) = make_files(tmp_path, "report.md")
    r = run_script("llmzip-compress.sh", src, env=env)
    assert r.returncode == 0, r.stderr
    lz = src.with_name("report.md.lz")
    assert lz.exists()
    run_script("llmzip-decompress.sh", lz, env=env)
    restored = src.with_name("report.restored.md")
    assert restored.read_bytes() == src.read_bytes()
    notes = notifications(tmp_path)
    assert notes[0][1].startswith("Compressing report.md")
    assert notes[1][0] == "llmzip ✓ compressed report.md" and "mode=LLM" in notes[1][1]
    assert "Decompressing report.md.lz" in notes[2][1]
    assert notes[3][0] == "llmzip ✓ restored report.md.lz"
    assert "Original vs restored: IDENTICAL ✓" in notes[3][1]
    log = (tmp_path / "finder.log").read_text(encoding="utf-8")
    assert "compress" in log and "-> exit 0" in log and "IDENTICAL" in log


@pytest.mark.slow
def test_spaces_apostrophes_and_no_overwrite(tmp_path, env):
    (src,) = make_files(tmp_path, "it's my notes.md")
    lz = src.with_name("it's my notes.md.lz")
    lz.write_bytes(b"existing archive")  # must not be overwritten
    restored = src.with_name("it's my notes.restored.md")
    restored.write_bytes(b"existing restored")  # must not be overwritten

    run_script("llmzip-compress.sh", src, env=env)
    second = src.with_name("it's my notes-2.md.lz")
    assert lz.read_bytes() == b"existing archive" and second.exists()

    run_script("llmzip-decompress.sh", second, env=env)
    # report-2.md.lz restores to "...-2.restored.md"; its original "it's my notes-2.md" is absent
    out = src.with_name("it's my notes-2.restored.md")
    assert out.read_bytes() == src.read_bytes()
    assert restored.read_bytes() == b"existing restored"
    last = notifications(tmp_path)[-1]
    assert "Original not present" in last[1]

    # decompressing the same archive again numbers the restored copy
    run_script("llmzip-decompress.sh", second, env=env)
    assert src.with_name("it's my notes-2.restored-2.md").read_bytes() == src.read_bytes()


@pytest.mark.slow
def test_decompress_rejects_non_lz_and_continues(tmp_path, env):
    plain, src = make_files(tmp_path, "plain.txt", "doc.md")
    subprocess.run([UV, "run", "--project", str(ROOT), "--frozen", "--quiet", "llmzip", "compress",
                    str(src), "--mode", "zstd", "--quiet"], check=True, env=env)
    r = run_script("llmzip-decompress.sh", plain, src.with_name("doc.md.lz"), env=env)
    assert r.returncode == 0
    notes = notifications(tmp_path)
    assert notes[0] == ("llmzip ✗ plain.txt", "Not a .lz file; only .lz files can be decompressed.")
    assert "IDENTICAL ✓" in notes[-1][1]
    assert src.with_name("doc.restored.md").read_bytes() == src.read_bytes()


@pytest.mark.slow
def test_compress_skips_folders_and_lz(tmp_path, env):
    (src,) = make_files(tmp_path, "a.md")
    folder = tmp_path / "work" / "sub dir"
    folder.mkdir()
    fake_lz = tmp_path / "work" / "old.md.lz"
    fake_lz.write_bytes(b"x")
    run_script("llmzip-compress.sh", folder, fake_lz, src, env=env)
    notes = notifications(tmp_path)
    assert "Skipped folder “sub dir”" in notes[0][1]
    assert "already a .lz file" in notes[1][1]
    assert notes[-1][0] == "llmzip ✓ compressed a.md"
    assert fake_lz.read_bytes() == b"x"


def test_compress_error_is_notified_and_next_file_continues(tmp_path, env):
    good = make_files(tmp_path, "good.md")[0]
    env = dict(env, LLMZIP_UV="/nonexistent/uv")  # every CLI run fails cleanly
    r = run_script("llmzip-compress.sh", tmp_path / "work" / "missing.md", good, env=env)
    assert r.returncode == 0
    notes = notifications(tmp_path)
    assert notes[0] == ("llmzip ✗ missing.md", "File not found.")
    assert notes[-1][0] == "llmzip ✗ good.md"


def test_inspect_dry_run_uses_absolute_path(tmp_path, env, monkeypatch):
    (src,) = make_files(tmp_path, "it's a “test”.md")
    rel = os.path.relpath(src, tmp_path)
    r = subprocess.run([str(FINDER / "llmzip-inspect.sh"), rel], cwd=tmp_path,
                       env=dict(env, LLMZIP_DRY_RUN="1"), capture_output=True, text=True)
    cmd = r.stdout.strip()
    assert r.returncode == 0 and cmd.startswith(f"cd '{ROOT}' && '{UV}' run --frozen llmzip tui ")
    assert cmd.endswith(" --tab heatmap")
    # the quoted path evaluates back to the absolute path
    echoed = subprocess.run(["zsh", "-c", "print -r -- " + cmd.split(" llmzip tui ", 1)[1].rsplit(" --tab", 1)[0]],
                            capture_output=True, text=True).stdout.strip()
    assert echoed == str(src.resolve())


def test_install_generates_valid_workflows(tmp_path):
    dest = tmp_path / "Services"
    r = subprocess.run([str(FINDER / "install.sh"), "--dest", str(dest)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    names = ["Compress with llmzip", "Decompress with llmzip", "Inspect with llmzip"]
    scripts = ["llmzip-compress.sh", "llmzip-decompress.sh", "llmzip-inspect.sh"]
    for name, script in zip(names, scripts):
        contents = dest / f"{name}.workflow" / "Contents"
        for plist in ("Info.plist", "document.wflow"):
            lint = subprocess.run(["plutil", "-lint", str(contents / plist)], capture_output=True, text=True)
            assert lint.returncode == 0 and lint.stdout.strip().endswith("OK")
        info = plistlib.loads((contents / "Info.plist").read_bytes())
        service = info["NSServices"][0]
        assert service["NSMenuItem"]["default"] == name and service["NSSendFileTypes"] == ["public.item"]
        doc = plistlib.loads((contents / "document.wflow").read_bytes())
        params = doc["actions"][0]["action"]["ActionParameters"]
        assert params["shell"] == "/bin/zsh" and params["inputMethod"] == 1
        cmd = params["COMMAND_STRING"]
        assert f"LLMZIP_REPO='{ROOT}'" in cmd and f"'{FINDER / script}'" in cmd and cmd.endswith('"$@"')
        assert "LLMZIP_UV='/" in cmd
        assert doc["workflowMetaData"]["workflowTypeIdentifier"] == "com.apple.Automator.servicesMenu"

    r = subprocess.run([str(FINDER / "install.sh"), "--dest", str(dest), "--uninstall"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and not any(dest.iterdir())
