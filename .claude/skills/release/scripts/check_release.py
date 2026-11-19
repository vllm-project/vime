#!/usr/bin/env python3
"""Check local vime release metadata, Docker pins and patch coverage."""

import argparse
import ast
import re
import shlex
import sys
from pathlib import Path


def _setup_version(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", None) != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "version":
                return ast.literal_eval(keyword.value)
    raise ValueError(f"setup version not found in {path}")


def _assigned_string(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{name} not found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--expected-version")
    args = parser.parse_args()

    repo = args.repo.resolve()
    errors: list[str] = []

    setup_version = _setup_version(repo / "setup.py")
    docs_version = _assigned_string(repo / "docs/conf.py", "__version__")
    if setup_version != docs_version:
        errors.append(f"setup.py={setup_version} but docs/conf.py={docs_version}")
    if args.expected_version and setup_version != args.expected_version:
        errors.append(f"release version is {setup_version}, expected {args.expected_version}")

    docker_text = (repo / "docker/Dockerfile").read_text()
    docker_args = {}
    for name, value in re.findall(r"^ARG ([A-Z][A-Z0-9_]*)=(\S+)$", docker_text, re.MULTILINE):
        if name in docker_args and docker_args[name] != value:
            errors.append(f"Docker stages disagree on {name}")
        docker_args[name] = value
        if name.endswith("_COMMIT") and name != "VIME_COMMIT" and not re.fullmatch(r"[0-9a-f]{40}", value):
            errors.append(f"{name} is not a full commit SHA: {value}")
    for name in ("VLLM_COMMIT", "MEGATRON_COMMIT", "PATCH_VERSION"):
        if name not in docker_args:
            errors.append(f"Docker is missing {name}")
    base_image = docker_args.get("BASE_IMAGE", "")
    if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", base_image):
        errors.append("BASE_IMAGE must use an immutable sha256 digest")

    patch_dir = repo / "docker/patch" / docker_args.get("PATCH_VERSION", "latest")
    patches = {path.name for path in patch_dir.glob("*.patch")}
    if not patches:
        errors.append(f"No patches found in {patch_dir}")
    used = set()
    for stage in re.split(r"^FROM ", docker_text.replace("\\\n", ""), flags=re.MULTILINE)[1:]:
        copied = set()
        for pattern in re.findall(r"^COPY docker/patch/\$\{PATCH_VERSION\}/(\S+)", stage, re.MULTILINE):
            matches = {path.name for path in patch_dir.glob(pattern)}
            if not matches:
                errors.append(f"Docker COPY has no matching patch: {pattern}")
            copied.update(matches)
        applied = {
            Path(token).name
            for command in re.findall(r"\bgit apply ([^&;\n]+)", stage)
            for token in shlex.split(command)
            if token.endswith(".patch")
        }
        if copied != applied:
            errors.append(
                f"Docker stage {stage.splitlines()[0]} patch coverage differs: "
                f"not_applied={sorted(copied - applied)}, not_copied={sorted(applied - copied)}"
            )
        used.update(copied)
    if patches != used:
        errors.append(f"Unused Docker patches: {sorted(patches - used)}")

    docker_version = (repo / "docker/version.txt").read_text().strip()
    if not re.fullmatch(r"nightly-dev-\d{8}[a-z]", docker_version):
        errors.append(f"unexpected docker/version.txt format: {docker_version}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        f"release={setup_version}, vllm={docker_args['VLLM_COMMIT']}, docker={docker_version}, patches={len(patches)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
