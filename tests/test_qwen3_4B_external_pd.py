"""E2E test for --rollout-external-engine-addrs with a pure-PD external fleet.

Spawns two vLLM servers out-of-band on a single GPU box (all tp=1):
- 1 prefill (NIXL ``kv_producer``)
- 1 decode  (NIXL ``kv_consumer``)

and points vime at both via ``--rollout-external-engine-addrs ...``.
The first 4 GPUs train. vime queries ``/server_info`` on each engine to
infer per-engine TP / GPU counts and registers them to its PD-enabled router.

Weight sync uses ``--update-weight-mode full --update-weight-transport disk``
so the post-train sync writes a complete HF checkpoint to a shared directory
and the external engines reload it through ``update_weights_from_disk`` without
forming an NCCL group with the trainer.
"""

import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import vime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
TORCH_DIST_CKPT = f"/root/models/{MODEL_NAME}_torch_dist"
NUM_GPUS = 6
NUM_TRAIN_GPUS = 4
NUM_PREFILL_ENGINES = 1
NUM_DECODE_ENGINES = 1

EXTERNAL_HOST = "127.0.0.1"
PREFILL_PORTS = [13150]
DECODE_PORTS = [13151]
BOOTSTRAP_PORTS = [13160]
DECODE_BOOTSTRAP_PORTS = [13161]


def _get_bond_ipv4():
    net_root = Path("/sys/class/net")
    if not net_root.exists():
        return None

    bond_ifaces = [
        path.name for path in net_root.iterdir() if path.name.startswith("bond") and path.name[4:].isdigit()
    ]
    bond_ifaces.sort(key=lambda name: int(name[4:]))
    for iface in bond_ifaces:
        try:
            output = subprocess.check_output(["ip", "-o", "-4", "addr", "show", "dev", iface], text=True)
        except (OSError, subprocess.CalledProcessError):
            continue
        fields = output.split()
        for idx, field in enumerate(fields):
            if field == "inet" and idx + 1 < len(fields):
                return fields[idx + 1].split("/", 1)[0]
    return None


def _get_external_host():
    env_value = os.environ.get("VIME_TEST_EXTERNAL_PD_HOST")
    if env_value and env_value not in ("127.0.0.1", "localhost"):
        return env_value

    bond_host = _get_bond_ipv4()
    if bond_host is not None:
        return bond_host

    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr and master_addr not in ("127.0.0.1", "localhost"):
        return master_addr

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
            if host and not host.startswith("127."):
                return host
    except OSError:
        pass

    return EXTERNAL_HOST


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_TRAIN_GPUS,
        dir_dst="/root/models",
    )


def _get_gpu_split():
    """Partition visible GPUs: 4 train + 1 prefill + 1 decode."""
    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(NUM_GPUS))).split(",")
    assert len(all_gpus) >= NUM_GPUS, f"Expected at least {NUM_GPUS} GPUs, got {len(all_gpus)}"
    train_gpus = all_gpus[:NUM_TRAIN_GPUS]
    cursor = NUM_TRAIN_GPUS
    prefill_gpus = all_gpus[cursor : cursor + NUM_PREFILL_ENGINES]
    cursor += NUM_PREFILL_ENGINES
    decode_gpus = all_gpus[cursor : cursor + NUM_DECODE_ENGINES]
    return train_gpus, prefill_gpus, decode_gpus


def _launch_vllm_server(
    *,
    gpus: list[str],
    port: int,
    tp: int,
    log_path: str,
    disaggregation_mode: str,
    disaggregation_bootstrap_port: int | None = None,
    external_host: str = EXTERNAL_HOST,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    assert disaggregation_bootstrap_port is not None
    env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = external_host
    env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(disaggregation_bootstrap_port)
    env["UCX_NET_DEVICES"] = os.environ.get("VIME_TEST_DISAGGREGATION_IB_DEVICE", "all").strip() or "all"

    kv_role = {
        "prefill": "kv_producer",
        "decode": "kv_consumer",
    }[disaggregation_mode]

    cmd = [
        "vllm",
        "serve",
        f"/root/models/{MODEL_NAME}",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--served-model-name",
        f"/root/models/{MODEL_NAME}",
        f"/root/models/{MODEL_NAME}/",
        "--tensor-parallel-size",
        str(tp),
        "--gpu-memory-utilization",
        "0.6",
        "--trust-remote-code",
        "--logprobs-mode",
        "processed_logprobs",
        "--enable-prompt-tokens-details",
        "--enable-server-load-tracking",
        "--kv-transfer-config",
        json.dumps({"kv_connector": "NixlConnector", "kv_role": kv_role}),
        "--weight-transfer-config",
        json.dumps({"backend": "nccl"}),
    ]

    log_file = open(log_path, "w")
    process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    print(
        f"Starting external vllm {disaggregation_mode} server on GPUs {gpus} "
        f"port={port} tp={tp} (pid={process.pid}), log: {log_path}"
    )

    # Wait up to ~10 minutes for /server_info to come up.  /health_generate
    # is unreliable for prefill/decode-only nodes, so we poll /server_info
    # — that's what vime's discover_external_engines uses anyway.
    deadline = time.time() + 600
    while time.time() < deadline:
        if process.poll() is not None:
            log_tail = Path(log_path).read_text(errors="replace")[-8000:]
            raise RuntimeError(
                f"{disaggregation_mode} server exited with code {process.returncode}; {log_path} tail:\n{log_tail}"
            )
        try:
            req = urllib.request.urlopen(f"http://{external_host}:{port}/server_info?config_format=json", timeout=2)
            if req.status == 200:
                print(f"External vllm {disaggregation_mode} server is ready on GPUs {gpus}")
                return process
        except Exception:
            pass
        time.sleep(5)

    process.kill()
    process.wait()
    log_tail = Path(log_path).read_text(errors="replace")[-8000:]
    raise RuntimeError(f"{disaggregation_mode} server failed to start within timeout; {log_path} tail:\n{log_tail}")


def execute():
    train_gpus, prefill_gpus, decode_gpus = _get_gpu_split()
    external_host = _get_external_host()
    print(f"Using external host for vLLM workers: {external_host}")
    processes: list[subprocess.Popen] = []

    # Restrict CUDA_VISIBLE_DEVICES to training GPUs before Ray starts so
    # ray's bundle allocator doesn't try to claim the external vllm GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpus)

    def launch_external_engines():
        for idx, (gpu, port, bootstrap_port) in enumerate(
            zip(prefill_gpus, PREFILL_PORTS, BOOTSTRAP_PORTS, strict=True)
        ):
            processes.append(
                _launch_vllm_server(
                    gpus=[gpu],
                    port=port,
                    tp=1,
                    disaggregation_mode="prefill",
                    disaggregation_bootstrap_port=bootstrap_port,
                    external_host=external_host,
                    log_path=f"/tmp/vllm_external_prefill_{idx}.log",
                )
            )
        for idx, (gpu, port, bootstrap_port) in enumerate(
            zip(decode_gpus, DECODE_PORTS, DECODE_BOOTSTRAP_PORTS, strict=True)
        ):
            processes.append(
                _launch_vllm_server(
                    gpus=[gpu],
                    port=port,
                    tp=1,
                    disaggregation_mode="decode",
                    disaggregation_bootstrap_port=bootstrap_port,
                    external_host=external_host,
                    log_path=f"/tmp/vllm_external_decode_{idx}.log",
                )
            )

    disk_dir_cm = tempfile.TemporaryDirectory(prefix="vime_external_pd_full_disk_")
    disk_dir = disk_dir_cm.name
    try:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load {TORCH_DIST_CKPT} "

        rollout_args = (
            "--prompt-data /root/datasets/gsm8k/train.parquet "
            "--input-key messages "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type math "
            "--num-rollout 3 "
            "--rollout-batch-size 4 "
            "--n-samples-per-prompt 4 "
            "--rollout-max-response-len 1024 "
            "--rollout-temperature 0.8 "
            "--global-batch-size 16 "
        )

        perf_args = (
            "--tensor-model-parallel-size 2 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--use-dynamic-batch-size "
            "--max-tokens-per-gpu 9216 "
            "--recompute-granularity full "
            "--recompute-method uniform "
            "--recompute-num-layers 1 "
        )

        grpo_args = (
            "--advantage-estimator grpo "
            "--use-kl-loss "
            "--kl-loss-coef 0.00 "
            "--kl-loss-type low_var_kl "
            # Nonzero entropy coef guarantees a nonzero gradient even when all
            # rewards in a group tie (advantages=0), so the delta sync writes
            # real sparse files instead of an empty no-op.
            "--entropy-coef 0.01 "
            "--eps-clip 0.2 "
            "--eps-clip-high 0.28 "
        )

        optimizer_args = (
            "--optimizer adam "
            "--lr 1e-6 "
            "--lr-decay-style constant "
            "--weight-decay 0.1 "
            "--adam-beta1 0.9 "
            "--adam-beta2 0.98 "
        )

        # No --rollout-num-gpus / --rollout-num-gpus-per-engine: those are
        # inferred from /server_info on each external engine (1 prefill +
        # 1 decode, all tp=1).
        all_addrs = [f"{external_host}:{port}" for port in (*PREFILL_PORTS, *DECODE_PORTS)]
        external_args = "--rollout-external-engine-addrs " + " ".join(all_addrs) + " "

        # External engines have no NCCL group with the trainer, so the trainer
        # publishes a complete HF checkpoint and the engines reload it from the
        # shared filesystem.
        disk_update_args = (
            "--update-weight-mode full "
            "--update-weight-transport disk "
            f"--update-weight-disk-dir {disk_dir} "
            "--update-weight-disk-keep-files "
        )

        ci_args = "--ci-test "

        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--actor-num-nodes 1 "
            f"--actor-num-gpus-per-node {NUM_TRAIN_GPUS} "
        )

        train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optimizer_args} "
            f"{grpo_args} "
            f"{U.get_default_wandb_args(__file__)} "
            f"{perf_args} "
            f"{external_args} "
            f"{disk_update_args} "
            f"{ci_args} "
            f"{misc_args} "
        )

        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_TRAIN_GPUS,
            megatron_model_type=MODEL_TYPE,
            before_ray_job_submit=launch_external_engines,
            extra_env_vars={
                "no_proxy": f"127.0.0.1,localhost,{external_host}",
                "NO_PROXY": f"127.0.0.1,localhost,{external_host}",
            },
        )

        checkpoint_dirs = sorted(Path(disk_dir).glob("weight_v*"))
        assert checkpoint_dirs, f"No disk checkpoint directories were written under {disk_dir}"
        assert any((path / "model.safetensors.index.json").exists() for path in checkpoint_dirs)
        assert any(list(path.glob("*.safetensors")) for path in checkpoint_dirs)
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
                p.wait()
        U.exec_command("pkill -9 vllm; true")
        disk_dir_cm.cleanup()


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
