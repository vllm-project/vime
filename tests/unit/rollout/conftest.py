"""Shared stubs for ``vime.rollout.vllm_rollout`` unit tests."""

from __future__ import annotations

import importlib.util
import sys
import types
from unittest.mock import MagicMock


def _real_module_available(name: str) -> bool:
    """True if the real module is importable, so we should NOT shadow it with a stub.

    These conftest stubs install bare ``sys.modules`` entries at import time and never
    clean up, so they leak to sibling test packages (e.g. ``tests/utils``). When the real
    dependency is installed (as in the CI image), a bare stub like ``vllm_router`` (no
    submodules) breaks unrelated imports such as ``from vllm_router.launch_router import``.
    Only stub when the real package is genuinely absent (bare-deps dev machines).
    """
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _ensure_vllm_router_stub() -> None:
    if _real_module_available("vllm_router"):
        return
    sys.modules["vllm_router"] = types.ModuleType("vllm_router")


def _ensure_pil_stub() -> None:
    if _real_module_available("PIL"):
        return
    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    pil.Image = image_mod
    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = image_mod


def _ensure_transformers_stub() -> None:
    if _real_module_available("transformers"):
        return
    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = type(
        "AutoTokenizer",
        (),
        {"from_pretrained": staticmethod(lambda *args, **kwargs: object())},
    )
    mod.AutoProcessor = type(
        "AutoProcessor",
        (),
        {"from_pretrained": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))},
    )
    mod.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    mod.ProcessorMixin = type("ProcessorMixin", (), {})
    sys.modules["transformers"] = mod


def _ensure_aiohttp_stub() -> None:
    if _real_module_available("aiohttp"):
        return
    sys.modules["aiohttp"] = MagicMock()


def _ensure_pylatexenc_stub() -> None:
    if _real_module_available("pylatexenc"):
        return
    pylatexenc = types.ModuleType("pylatexenc")
    latex2text = types.ModuleType("pylatexenc.latex2text")
    pylatexenc.latex2text = latex2text
    sys.modules["pylatexenc"] = pylatexenc
    sys.modules["pylatexenc.latex2text"] = latex2text


_ensure_vllm_router_stub()
_ensure_pil_stub()
_ensure_transformers_stub()
_ensure_aiohttp_stub()
_ensure_pylatexenc_stub()
