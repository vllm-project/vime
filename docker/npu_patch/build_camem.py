"""Rebuild only the patched CANN allocator, preserving prebuilt Ascend ops."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sysconfig
import tempfile
from pathlib import Path


def build(source_root: Path, cann_root: Path) -> Path:
    source = source_root / "csrc/camem_allocator.cpp"
    include = cann_root / "include"
    library = cann_root / "lib64"
    # CANN's installed toolkit exposes the public headers through include/.
    header = include / "acl/acl.h"
    for path in (source, header):
        if not path.is_file():
            raise FileNotFoundError(f"CaMem build input missing: {path}")
    output = source_root / "vllm_ascend" / ("vime_camem_allocator" + sysconfig.get_config_var("EXT_SUFFIX"))
    stamp = output.with_suffix(output.suffix + ".json")
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    flags = [
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        "-pthread",
        "-DVIME_CAMEM_STANDALONE=1",
        "-Wl,-Bsymbolic",
        "-I" + sysconfig.get_path("include"),
        "-I" + str(include),
        "-L" + str(library),
        "-Wl,-rpath," + str(library),
    ]
    inputs = {
        "source": hashlib.sha256(source.read_bytes()).hexdigest(),
        "headers": {
            str(path.relative_to(include)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((include / "acl").rglob("*.h"))
        },
        "soabi": sysconfig.get_config_var("SOABI"),
        "compiler": subprocess.check_output(compiler + ["--version"], text=True),
        "flags": flags,
    }
    if output.is_file() and stamp.is_file() and json.loads(stamp.read_text()) == inputs:
        print(f"CaMem native allocator unchanged: {output}", flush=True)
        return output
    # Same filesystem for atomic replacement; no persistent build intermediates.
    with tempfile.TemporaryDirectory(prefix=".camem-build-", dir=output.parent) as directory:
        built = Path(directory) / output.name
        subprocess.run(compiler + flags + [str(source), "-lascendcl", "-o", str(built)], check=True)
        built.replace(output)
    stamp.write_text(json.dumps(inputs, sort_keys=True))
    print(f"Built patched CaMem native allocator: {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/vllm-workspace/vllm-ascend"))
    parser.add_argument(
        "--cann-root",
        type=Path,
        default=Path(
            os.environ.get("ASCEND_HOME_PATH")
            or os.environ.get("ASCEND_TOOLKIT_HOME")
            or "/usr/local/Ascend/ascend-toolkit/latest"
        ),
    )
    build(**vars(parser.parse_args()))
