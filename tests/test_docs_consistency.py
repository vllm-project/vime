"""Guard documentation references that can be checked without a GPU runtime."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

NUM_GPUS = 0

ROOT = Path(__file__).resolve().parents[1]
DOC_SOURCES = (
    ROOT / "README.md",
    ROOT / "README_zh.md",
    ROOT / "docs" / "en",
    ROOT / "docs" / "zh",
    ROOT / "examples",
    ROOT / "docker",
)
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
COMMAND_PATH_RE = re.compile(r"\b(?:bash|python)\s+((?:scripts?|tests)/[A-Za-z0-9_.+/-]+\.(?:sh|py))")
LOCAL_ANCHOR_RE = re.compile(r"\]\(#([A-Za-z0-9_-]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
RELEASE_CHECK = ROOT / ".claude/skills/release/scripts/check_release.py"


def _markdown_files():
    for source in DOC_SOURCES:
        if source.is_file():
            yield source
        else:
            for path in sorted(source.rglob("*.md")):
                if "_examples_synced" not in path.parts:
                    yield path


def _local_link_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]

    if target.startswith(("#", "/")) or "://" in target or target.startswith(("mailto:", "data:")):
        return None

    return unquote(target.split("#", 1)[0].split("?", 1)[0]) or None


def test_local_markdown_links_exist():
    missing = []
    for markdown_file in _markdown_files():
        for raw_target in MARKDOWN_LINK_RE.findall(markdown_file.read_text(encoding="utf-8")):
            target = _local_link_target(raw_target)
            if (
                target is not None
                and "_examples_synced" not in Path(target).parts
                and not (markdown_file.parent / target).exists()
            ):
                missing.append(f"{markdown_file.relative_to(ROOT)} -> {target}")

    assert not missing, "Broken local Markdown links:\n" + "\n".join(missing)


def test_documented_script_and_test_commands_exist():
    missing = []
    for markdown_file in _markdown_files():
        text = markdown_file.read_text(encoding="utf-8")
        for command_path in COMMAND_PATH_RE.findall(text):
            if not (ROOT / command_path).is_file():
                missing.append(f"{markdown_file.relative_to(ROOT)} -> {command_path}")

    assert not missing, "Documented command paths do not exist:\n" + "\n".join(missing)


@pytest.mark.parametrize("language", ["en", "zh"])
def test_customization_anchor_links_exist(language):
    text = (ROOT / "docs" / language / "get_started" / "customization.md").read_text(encoding="utf-8")
    headings = set()
    for heading in HEADING_RE.findall(text):
        heading = re.sub(r"[^\w\s-]", "", heading.replace("`", "").lower())
        headings.add(re.sub(r"\s+", "-", heading).strip("-"))

    missing = sorted(set(LOCAL_ANCHOR_RE.findall(text)) - headings)
    assert not missing, f"Local anchors without matching headings: {missing}"


def test_release_metadata_check():
    result = subprocess.run([sys.executable, str(RELEASE_CHECK), "--repo", str(ROOT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def release_repo(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docker/patch/latest").mkdir(parents=True)
    (tmp_path / "setup.py").write_text('setup(version="1.2.3")\n')
    (tmp_path / "docs/conf.py").write_text('__version__ = "1.2.3"\n')
    (tmp_path / "docker/version.txt").write_text("nightly-dev-20261012a\n")
    (tmp_path / "docker/patch/latest/vllm.patch").write_text("")
    (tmp_path / "docker/Dockerfile").write_text(
        "ARG BASE_IMAGE=vllm/vllm-openai@sha256:" + "a" * 64 + "\n"
        "FROM ${BASE_IMAGE}\nARG PATCH_VERSION=latest\n"
        "ARG VLLM_COMMIT=" + "b" * 40 + "\n"
        "ARG MEGATRON_COMMIT=" + "c" * 40 + "\n"
        "COPY docker/patch/${PATCH_VERSION}/vllm.patch /tmp/vllm.patch\n"
        "RUN git apply --include='rust/*' /tmp/vllm.patch\n"
    )
    return tmp_path


@pytest.mark.parametrize(
    ("path", "old", "new", "error"),
    [
        ("docs/conf.py", "1.2.3", "1.2.4", "docs/conf.py"),
        ("docker/Dockerfile", "@sha256:" + "a" * 64, ":nightly", "immutable"),
        ("docker/Dockerfile", "b" * 40, "main", "full commit SHA"),
        ("docker/Dockerfile", "RUN git apply", "RUN echo", "not_applied"),
        ("docker/Dockerfile", "ARG PATCH_VERSION=latest", "ARG PATCH_VERSION=absent", "No patches"),
        ("docker/version.txt", "nightly-dev-20261012a", "latest", "format"),
    ],
)
def test_release_metadata_rejects_invalid_inputs(release_repo, path, old, new, error):
    target = release_repo / path
    target.write_text(target.read_text().replace(old, new))
    result = subprocess.run(
        [sys.executable, str(RELEASE_CHECK), "--repo", str(release_repo)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert error in result.stderr


def test_release_metadata_rejects_orphan_patch(release_repo):
    (release_repo / "docker/patch/latest/orphan.patch").write_text("")
    result = subprocess.run(
        [sys.executable, str(RELEASE_CHECK), "--repo", str(release_repo)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "orphan.patch" in result.stderr


@pytest.mark.parametrize(("version", "status"), [("1.2.3", 0), ("1.2.4", 1)])
def test_release_metadata_expected_version(release_repo, version, status):
    result = subprocess.run(
        [sys.executable, str(RELEASE_CHECK), "--repo", str(release_repo), "--expected-version", version],
        capture_output=True,
        text=True,
    )
    assert result.returncode == status, result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
