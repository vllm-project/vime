"""Shared stubs for ``vime.rollout.vllm_rollout`` unit tests."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _ensure_vllm_router_stub() -> None:
    if "vllm_router" in sys.modules:
        return
    sys.modules["vllm_router"] = types.ModuleType("vllm_router")


def _ensure_pil_stub() -> None:
    if "PIL" in sys.modules:
        return
    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    pil.Image = image_mod
    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = image_mod


def _ensure_transformers_stub() -> None:
    if "transformers" in sys.modules:
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
    if "aiohttp" in sys.modules:
        return
    sys.modules["aiohttp"] = MagicMock()


def _ensure_pylatexenc_stub() -> None:
    if "pylatexenc" in sys.modules:
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
