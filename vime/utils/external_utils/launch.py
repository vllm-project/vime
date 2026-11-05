"""Device-specific launch policy for the vime test/example harness.

Used only by the test and example launch utilities (`command_utils.execute_train`), not by
vime core: a `Platform` describes one accelerator for the purpose of *launching a job* — how
Ray advertises its devices, the device runtime env, and how to construct the launch command.
It adapts the core ``vime.platforms`` selection to shell commands.

Imports stay stdlib-only (torch imports are lazy) so the module is unit-testable in isolation;
the actual `exec_command` calls live in `command_utils`.
"""

import json
import shlex
from dataclasses import dataclass, field


# ── Platform contract ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Platform:
    name: str
    ray_args: str  # ray-start resource flags, "{n}"-templated with the device count
    env: dict = field(default_factory=dict)  # device runtime env (into runtime_env + raylet)

    def ray_start_args(self, num_devices: int) -> str:
        return self.ray_args.format(n=num_devices)


# ── Registry ────────────────────────────────────────────────────────────────

PLATFORMS: dict[str, Platform] = {}


def register(platform: Platform) -> None:
    PLATFORMS[platform.name] = platform


# ── Registered platforms (add a new accelerator here) ─────────────────────────


register(Platform(name="cuda", ray_args="--num-gpus {n}"))  # default; other fields unused for cuda

register(
    Platform(
        name="npu",
        # vime requests NPU bundles, not GPU (see ray/placement_group.py), so advertise
        # the custom NPU resource rather than Ray GPU capacity.
        ray_args="--num-gpus 0 --resources '{{\"NPU\": {n}}}'",
        env={
            "PYTHONPATH": (
                "/root/Megatron-LM:/root/vime:"
                "/root/MegatronAdaptor:/root/TransformerEngineNPU:"
                "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:"
                "/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge"
            ),
            "LD_LIBRARY_PATH": (
                "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:"
                "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64:"
                "/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:"
                "/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:"
                "/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/:"
                "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:"
                "/usr/local/Ascend/cann/lib64:/usr/local/Ascend/cann/aarch64-linux/devlib"
            ),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
            "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
            "HCCL_CONNECT_TIMEOUT": "7200",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "HYDRA_FULL_ERROR": "1",
            "TRANSFORMERS_VERBOSITY": "error",  # silence transformers image-processing log spam
        },
    )
)


# ── Resolver + launcher (platform-agnostic; unchanged when adding a platform) ──


def current_platform() -> Platform:
    """Adapt the selected core runtime platform to the launch contract."""
    from vime.platforms import current_platform as current_runtime_platform

    selected = current_runtime_platform().name
    try:
        return PLATFORMS[selected]
    except KeyError as exc:
        raise ValueError(f"No launcher adapter is registered for platform {selected!r}") from exc


def launch_commands(
    platform: Platform,
    train_args: str,
    num_devices: int,
    megatron_model_type: str | None,
    repo_base_dir: str,
    extra_env: dict | None = None,
    external_ray: bool = False,
    master_addr: str = "127.0.0.1",
    ray_running: bool = False,
) -> list:
    """Ordered shell commands to launch a training job on `platform` (pure; caller execs).

    Driven entirely by platform data (ray resources + device env), so any non-CUDA platform
    works with no change here. The caller checks Ray liveness; existing clusters
    and independently owned serving processes must survive trainer restarts.
    """
    extra_env = extra_env or {}
    all_env = {**platform.env, **extra_env}
    all_env["VIME_PLATFORM"] = platform.name
    cmds: list = []
    if not external_ray and not ray_running:
        # Export the device env BEFORE `ray start` so the raylet and the actors it spawns
        # (e.g. vLLM engines) inherit it; a job's runtime_env does not reach those actors.
        exports = "".join(f"export {k}={shlex.quote(str(v))} && " for k, v in all_env.items())
        cmds.append(
            f"{exports}export PYTHONUNBUFFERED=1 && ray start --head --node-ip-address {master_addr} "
            f"{platform.ray_start_args(num_devices)} --disable-usage-stats"
        )

    runtime_env = {"env_vars": {"no_proxy": f"127.0.0.1,{master_addr}", "MASTER_ADDR": master_addr, **all_env}}
    src = f'source "{repo_base_dir}/scripts/models/{megatron_model_type}.sh" && ' if megatron_model_type else ""
    model_args = "${MODEL_ARGS[@]}" if megatron_model_type else ""
    cmds.append(
        f"export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && {src}"
        f'ray job submit --address="http://127.0.0.1:8265" '
        f"--runtime-env-json={shlex.quote(json.dumps(runtime_env))} "
        f"-- python3 train.py {model_args} {train_args}"
    )
    return cmds
