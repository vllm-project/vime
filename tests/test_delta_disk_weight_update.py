"""E2E smoke test for XOR and overwrite delta weight updates through disk.

Runs the full disk test configuration for both delta encodings. Two rollout
rounds exercise inference after a delta reload; recovered tensors are checked
against the published checksums.
"""

import json
import os
import shlex
import tempfile
from pathlib import Path

from safetensors import safe_open

import vime.utils.external_utils.command_utils as U
from vime.utils.disk_delta import checksum, make_tensor_reader


MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 4
TEST_ROOT = os.environ.get("HF_HOME") or "/root"
HF_CKPT = f"{TEST_ROOT}/models/{MODEL_NAME}"
DATASET_DIR = f"{TEST_ROOT}/datasets/gsm8k"
TORCH_DIST_CKPT = f"/dev/shm/{MODEL_NAME}_torch_dist"


def prepare():
    U.exec_command(f"mkdir -p {shlex.quote(HF_CKPT)} {shlex.quote(DATASET_DIR)}")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {shlex.quote(HF_CKPT)}")
    U.exec_command("hf download --repo-type dataset zhuzilin/gsm8k " f"--local-dir {shlex.quote(DATASET_DIR)}")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/dev/shm",
        hf_checkpoint=HF_CKPT,
    )


def verify_checkpoints(case_dir: Path, encoding: str):
    published = case_dir / "published"
    local = case_dir / "local"
    indexes = []
    for version in (1, 2):
        path = published / f"weight_v{version:06d}"
        index = json.loads((path / "model.safetensors.index.json").read_text())
        metadata = index["metadata"]
        assert metadata["version"] == f"{version:06d}", metadata
        assert metadata["base_version"] == f"{version - 1:06d}", metadata
        assert metadata["delta_encoding"] == encoding, metadata
        assert metadata["compression_format"] == "zstd", metadata
        assert metadata["checksum_format"] == "adler32", metadata
        assert index["weight_map"], f"No changed tensors in delta version {version}"
        for filename in set(index["weight_map"].values()):
            assert (path / filename).stat().st_size > 0
        indexes.append(index)

    state = json.loads((local / ".weight_sync/state.json").read_text())
    assert state == {"version": "000002"}, state
    read_tensor = make_tensor_reader(str(local))
    final_index = indexes[-1]
    checked = 0
    for filename in set(final_index["weight_map"].values()):
        with safe_open(str(published / "weight_v000002" / filename), framework="np") as delta:
            digests = delta.metadata()
            for name in delta.keys():
                assert final_index["weight_map"][name] == filename
                actual = checksum("adler32", read_tensor(name))
                assert actual == digests[name], f"Recovered weight checksum mismatch: {name}"
                checked += 1
    assert checked == len(final_index["weight_map"])
    summary = {
        "encoding": encoding,
        "rollout_rounds": 2,
        "local_version": state["version"],
        "checked_tensors": checked,
    }
    (case_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Disk delta validation passed: {summary}", flush=True)


def execute():
    for encoding in ("xor", "overwrite"):
        with tempfile.TemporaryDirectory(prefix=f"vime_delta_disk_{encoding}_") as disk_dir:
            ref_load = TORCH_DIST_CKPT if U.current_platform().torch_dist_convert else HF_CKPT
            ckpt_args = f"--hf-checkpoint {shlex.quote(HF_CKPT)} " f"--ref-load {shlex.quote(ref_load)} "

            rollout_args = (
                f"--prompt-data {shlex.quote(f'{DATASET_DIR}/train.parquet')} "
                "--input-key messages "
                "--label-key label "
                "--apply-chat-template "
                "--rollout-shuffle "
                "--rm-type math "
                "--num-rollout 2 "
                "--rollout-batch-size 4 "
                "--n-samples-per-prompt 4 "
                "--rollout-max-response-len 1024 "
                "--rollout-temperature 0.8 "
                "--over-sampling-batch-size 8 "
                "--dynamic-sampling-filter-path vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
                "--global-batch-size 16 "
            )

            perf_args = (
                "--tensor-model-parallel-size 1 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 1 "
                "--expert-tensor-parallel-size 1 "
                "--use-dynamic-batch-size "
                "--max-tokens-per-gpu 9216 "
            )

            grpo_args = (
                "--advantage-estimator grpo "
                "--use-kl-loss "
                "--kl-loss-coef 0.00 "
                "--kl-loss-type low_var_kl "
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

            vllm_args = (
                "--rollout-num-gpus-per-engine 1 "
                "--rollout-num-gpus 3 "
                "--vllm-gpu-memory-utilization 0.7 "
                "--vllm-max-cudagraph-capture-size 32 "
            )

            disk_update_args = (
                "--update-weight-mode delta "
                "--update-weight-transport disk "
                f"--update-weight-disk-dir {disk_dir}/published "
                f"--update-weight-local-checkpoint-dir {disk_dir}/local "
                f"--update-weight-delta-encoding {encoding} "
                "--update-weight-delta-checksum adler32 "
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
                "--actor-num-gpus-per-node 1 "
            )

            train_args = (
                f"{ckpt_args} "
                f"{rollout_args} "
                f"{optimizer_args} "
                f"{grpo_args} "
                f"{U.get_default_wandb_args(__file__)} "
                f"{perf_args} "
                f"{vllm_args} "
                f"{disk_update_args} "
                f"{ci_args} "
                f"{misc_args} "
            )

            U.execute_train(
                train_args=train_args,
                num_gpus_per_node=NUM_GPUS,
                megatron_model_type=MODEL_TYPE,
            )

            verify_checkpoints(Path(disk_dir), encoding)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
