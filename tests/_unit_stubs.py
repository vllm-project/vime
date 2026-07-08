"""Shared import stubs so CPU-only unit tests can import production modules.

Why this module exists
----------------------
The CPU CI image and many dev machines do not ship the full training stack
(Megatron, Ray, vLLM, transformers, Triton, …). Production code still imports
those packages at module load time. These helpers stub ``sys.modules`` so
tests can exercise vime logic without installing every optional dependency.

Stubs must not leak across pytest collection: install them in the test file
(or inside a fixture) immediately before the import under test, and restore
on teardown when sibling modules need the real package.

Patterns
--------
* **Optional deps** (``install_rollout_optional_stubs``, ``install_vllm_cli_stubs``):
  stub only when the real package is absent — safe at the top of a test file.
* **Scoped stubs** (``save_sys_modules`` / ``restore_sys_modules``): pop, install,
  import, then restore inside a module-scoped fixture so collection stays clean.

Import via ``import _unit_stubs`` after prepending ``tests/`` to ``sys.path``
(each test file bootstraps this; CI runs ``python tests/…`` or ``python tests/utils/…``).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock


def real_module_available(name: str) -> bool:
    """True when the real package is importable and should not be shadowed."""
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def ensure_ray_stub() -> None:
    if real_module_available("ray"):
        return
    ray = MagicMock()
    sys.modules["ray"] = ray
    sys.modules["ray._private"] = MagicMock()
    sys.modules["ray._private.services"] = MagicMock()
    sys.modules["ray.actor"] = MagicMock()


def install_rollout_optional_stubs() -> None:
    """Stub rollout-side optional imports when not installed."""
    ensure_ray_stub()

    install_vllm_router_stub()

    if not real_module_available("PIL"):
        pil = types.ModuleType("PIL")
        image_mod = types.ModuleType("PIL.Image")
        pil.Image = image_mod
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = image_mod

    if not real_module_available("transformers"):

        def _raise_os_error(*args, **kwargs):
            raise OSError()

        mod = types.ModuleType("transformers")
        mod.AutoTokenizer = type(
            "AutoTokenizer",
            (),
            {"from_pretrained": staticmethod(lambda *args, **kwargs: object())},
        )
        mod.AutoProcessor = type(
            "AutoProcessor",
            (),
            {"from_pretrained": staticmethod(_raise_os_error)},
        )
        mod.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
        mod.ProcessorMixin = type("ProcessorMixin", (), {})
        sys.modules["transformers"] = mod

    if not real_module_available("aiohttp"):
        sys.modules["aiohttp"] = MagicMock()

    if not real_module_available("pylatexenc"):
        pylatexenc = types.ModuleType("pylatexenc")
        latex2text = types.ModuleType("pylatexenc.latex2text")
        pylatexenc.latex2text = latex2text
        sys.modules["pylatexenc"] = pylatexenc
        sys.modules["pylatexenc.latex2text"] = latex2text

    install_wandb_stub()


def install_vllm_router_stub() -> None:
    if real_module_available("vllm_router"):
        return

    class RouterArgs:
        @classmethod
        def add_cli_args(cls, parser, *args, **kwargs):  # noqa: ARG003
            return parser

        @classmethod
        def from_cli_args(cls, args, *unused_args, **unused_kwargs):  # noqa: ARG003
            return types.SimpleNamespace()

    router_mod = types.ModuleType("vllm_router")
    router_mod.__path__ = []
    launch_router_mod = types.ModuleType("vllm_router.launch_router")
    router_args_mod = types.ModuleType("vllm_router.router_args")
    launch_router_mod.RouterArgs = RouterArgs
    router_args_mod.RouterArgs = RouterArgs
    router_mod.launch_router = launch_router_mod
    router_mod.router_args = router_args_mod
    sys.modules["vllm_router"] = router_mod
    sys.modules["vllm_router.launch_router"] = launch_router_mod
    sys.modules["vllm_router.router_args"] = router_args_mod


def install_wandb_stub() -> None:
    if real_module_available("wandb"):
        return
    wandb_mod = types.ModuleType("wandb")
    wandb_mod.run = None
    wandb_mod.log = MagicMock()
    wandb_mod.finish = MagicMock()
    wandb_mod.login = MagicMock()
    wandb_mod.init = MagicMock()
    wandb_mod.Settings = MagicMock()
    wandb_mod.util = types.SimpleNamespace(generate_id=lambda: "unit-test")
    sys.modules["wandb"] = wandb_mod


def save_sys_modules(names: Iterable[str]) -> dict[str, Any]:
    return {k: sys.modules.get(k) for k in names}


def restore_sys_modules(saved: dict[str, Any]) -> None:
    for k, original in saved.items():
        if original is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = original


@contextmanager
def isolated_sys_modules(names: Iterable[str]):
    saved = save_sys_modules(names)
    for k in names:
        sys.modules.pop(k, None)
    try:
        yield
    finally:
        restore_sys_modules(saved)


def install_megatron_mpu_stub() -> MagicMock:
    """Stub ``megatron.core.mpu`` (and minimal submodules) when Megatron is absent."""
    mpu_stub = MagicMock()
    mpu_stub.get_data_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_world_size.return_value = 2
    mpu_stub.get_tensor_model_parallel_group.return_value = "tp_group"
    mpu_stub.get_pipeline_model_parallel_rank.return_value = 0
    mpu_stub.get_pipeline_model_parallel_world_size.return_value = 1
    mpu_stub.get_expert_model_parallel_world_size.return_value = 1
    mpu_stub.get_expert_model_parallel_group.return_value = "ep_group"

    megatron_core = types.ModuleType("megatron.core")
    megatron_core.__path__ = []
    megatron_core.mpu = mpu_stub
    parallel_state_mod = types.ModuleType("megatron.core.parallel_state")
    parallel_state_mod.get_tensor_model_parallel_rank = mpu_stub.get_tensor_model_parallel_rank
    parallel_state_mod.get_tensor_model_parallel_world_size = mpu_stub.get_tensor_model_parallel_world_size
    transformer_mod = types.ModuleType("megatron.core.transformer")
    transformer_mod.__path__ = []
    transformer_layer_mod = types.ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer_mod.get_transformer_layer_offset = lambda *args, **kwargs: 0
    transformer_mod.transformer_layer = transformer_layer_mod
    megatron_core.parallel_state = parallel_state_mod
    megatron_core.transformer = transformer_mod
    megatron_mod = types.ModuleType("megatron")
    megatron_mod.core = megatron_core
    sys.modules.setdefault("megatron", megatron_mod)
    sys.modules.setdefault("megatron.core", megatron_core)
    sys.modules.setdefault("megatron.core.parallel_state", parallel_state_mod)
    sys.modules.setdefault("megatron.core.transformer", transformer_mod)
    sys.modules.setdefault("megatron.core.transformer.transformer_layer", transformer_layer_mod)
    return mpu_stub


def install_ray_stub() -> None:
    ray_mod = types.ModuleType("ray")
    ray_mod.get = lambda refs: refs
    ray_mod.ObjectRef = object
    ray_mod.actor = types.ModuleType("ray.actor")
    ray_mod.actor.ActorHandle = object
    ray_mod._private = types.SimpleNamespace(services=types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1"))
    sys.modules.setdefault("ray", ray_mod)
    sys.modules.setdefault("ray.actor", ray_mod.actor)


def install_vllm_cli_stubs() -> None:
    """Stub vLLM CLI/parser imports for ``vime.backends.vllm_utils.arguments`` when vLLM is absent."""
    if real_module_available("vllm"):
        return

    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__path__ = []

    utils_mod = types.ModuleType("vllm.utils")
    argparse_utils = types.ModuleType("vllm.utils.argparse_utils")

    import argparse

    class FlexibleArgumentParser(argparse.ArgumentParser):
        pass

    argparse_utils.FlexibleArgumentParser = FlexibleArgumentParser
    utils_mod.argparse_utils = argparse_utils

    engine_mod = types.ModuleType("vllm.engine")
    engine_mod.__path__ = []
    arg_utils = types.ModuleType("vllm.engine.arg_utils")

    class AsyncEngineArgs:
        @classmethod
        def add_cli_args(cls, parser):  # noqa: ARG003
            return parser

    arg_utils.AsyncEngineArgs = AsyncEngineArgs
    engine_mod.arg_utils = arg_utils
    system_utils_mod = types.ModuleType("vllm.utils.system_utils")
    system_utils_mod.kill_process_tree = lambda pid, include_parent=True: None  # noqa: ARG005
    utils_mod.system_utils = system_utils_mod

    # vllm.entrypoints stubs (used by arguments.add_vllm_arguments and vllm_engine._vllm_server_field_names)
    entrypoints_mod = types.ModuleType("vllm.entrypoints")
    entrypoints_mod.__path__ = []
    openai_mod = types.ModuleType("vllm.entrypoints.openai")
    openai_mod.__path__ = []
    cli_args_mod = types.ModuleType("vllm.entrypoints.openai.cli_args")

    import dataclasses as _dc

    @_dc.dataclass
    class FrontendArgs:
        @classmethod
        def add_cli_args(cls, parser):  # noqa: ARG003
            return parser

    cli_args_mod.FrontendArgs = FrontendArgs
    cli_args_mod.make_arg_parser = lambda parser=None: parser
    cli_args_mod.validate_parsed_serve_args = lambda args: args
    openai_mod.cli_args = cli_args_mod
    entrypoints_mod.openai = openai_mod
    vllm_mod.entrypoints = entrypoints_mod

    cli_mod = types.ModuleType("vllm.entrypoints.cli")
    cli_mod.__path__ = []
    serve_mod = types.ModuleType("vllm.entrypoints.cli.serve")

    class ServeSubcommand:
        pass

    serve_mod.ServeSubcommand = ServeSubcommand
    cli_mod.serve = serve_mod
    entrypoints_mod.cli = cli_mod

    vllm_mod.engine = engine_mod
    vllm_mod.utils = utils_mod

    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.utils"] = utils_mod
    sys.modules["vllm.utils.argparse_utils"] = argparse_utils
    sys.modules["vllm.utils.system_utils"] = system_utils_mod
    sys.modules["vllm.engine"] = engine_mod
    sys.modules["vllm.engine.arg_utils"] = arg_utils
    sys.modules["vllm.entrypoints"] = entrypoints_mod
    sys.modules["vllm.entrypoints.openai"] = openai_mod
    sys.modules["vllm.entrypoints.openai.cli_args"] = cli_args_mod
    sys.modules["vllm.entrypoints.cli"] = cli_mod
    sys.modules["vllm.entrypoints.cli.serve"] = serve_mod


def install_triton_stub() -> None:
    # CPU CI images may have a broken/partial triton install that imports
    # but fails during module init, so always override it for unit tests.
    triton_mod = MagicMock()
    triton_mod.jit = lambda fn: fn
    triton_mod.cdiv = lambda a, b: (a + b - 1) // b
    triton_mod.next_power_of_2 = lambda x: x
    triton_mod.__version__ = "0.0.0"
    language = MagicMock()
    triton_mod.language = language
    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = language


def install_vime_distributed_utils_stub() -> None:
    vime_utils = types.ModuleType("vime.utils.distributed_utils")
    vime_utils.get_gloo_group = MagicMock(return_value="gloo")
    sys.modules.setdefault("vime.utils.distributed_utils", vime_utils)
