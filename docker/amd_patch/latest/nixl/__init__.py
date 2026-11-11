# Ray's NIXL tensor transport imports `nixl`; the ROCm wheel (nixl-rocm) ships the same API as `nixl_rocm`.
import importlib
import sys

from nixl_rocm import *  # noqa: F401,F403

for _name in ("_api", "_bindings", "_utils", "logging"):
    _module = importlib.import_module(f"nixl_rocm.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
