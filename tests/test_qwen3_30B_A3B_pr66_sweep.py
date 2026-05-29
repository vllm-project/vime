"""PR #66 multi-node vLLM rollout topology sweep.

Drives the same test through ~10 parallelism configurations, picked by the
``PR66_CONFIG`` env var. Configs ``ref``, ``s1``, ``s2``, ``s6``, ``s8``, ``s10``
are single-host; ``s3``, ``s4``, ``s5``, ``s7``, ``s9`` are cross-host and require
the caller to build the Ray cluster externally and set
``SLIME_SCRIPT_EXTERNAL_RAY=1`` + ``MASTER_ADDR=<head_ip>`` (see ``head.sh`` /
``worker.sh`` next to this file).

Each config exercises a different combination of the vLLM rollout topology axes
PR #66 introduced (``nnodes_per_engine``, ``--headless`` on non-leader ranks,
mp backends when ``nnodes > 1``) and the existing Megatron parallelism axes
(TP / PP / CP / EP).
"""

import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"

# (actor_nodes, actor_gpus_per_node, rollout_gpus, engine_gpus,
#  num_gpus_per_node, mode, mt_tp, mt_pp, mt_cp, mt_ep, vllm_pp)
CONFIGS = {
    # Single-host colocate baselines (PR #66 with nnodes==1)
    "ref": (1, 8, 8, 8, 8, "colocate", 4, 1, 2, 8, 1),
    "s1":  (1, 8, 8, 4, 8, "colocate", 4, 1, 2, 8, 1),
    "s2":  (1, 8, 8, 2, 8, "colocate", 4, 1, 2, 8, 1),
    # Cross-host colocate (PR #66 multi-node vLLM, nnodes>1)
    "s3":  (2, 8, 16, 16, 8, "colocate", 8, 1, 2, 8, 1),
    "s4":  (2, 8, 16, 8, 4, "colocate", 8, 1, 2, 8, 1),
    "s5":  (2, 8, 16, 8, 8, "colocate", 8, 1, 2, 8, 1),
    # Disagg (train / rollout on disjoint GPU pools)
    "s6":  (1, 4, 4, 4, 4, "disagg",   4, 1, 1, 4, 1),
    "s7":  (1, 8, 8, 8, 8, "disagg",   4, 1, 2, 8, 1),
    # vLLM PP > 1 — exercises _get_vllm_tp_size's pp-divisor path
    "s8":  (1, 8, 8, 8, 8, "colocate", 4, 1, 2, 8, 2),
    "s9":  (2, 8, 16, 16, 8, "colocate", 8, 1, 2, 8, 2),
    # Megatron PP > 1 — exercises the pipeline-stage weight-iterator path
    "s10": (1, 8, 8, 8, 8, "colocate", 4, 2, 1, 4, 1),
}


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=8,
        dir_dst="/root/models",
    )


def execute():
    cfg_id = os.environ.get("PR66_CONFIG", "ref")
    (actor_nodes, actor_gpus_per_node, rollout_gpus, engine_gpus,
     num_gpus_per_node, mode, mt_tp, mt_pp, mt_cp, mt_ep, vllm_pp) = CONFIGS[cfg_id]

    nnodes_per_engine = max(1, engine_gpus // num_gpus_per_node)
    print(
        f"[PR66 sweep] config={cfg_id}: actor={actor_nodes}n*{actor_gpus_per_node}g, "
        f"rollout={rollout_gpus}g (engine={engine_gpus}g, nnodes_per_engine={nnodes_per_engine}), "
        f"num_gpus_per_node={num_gpus_per_node}, mode={mode}, "
        f"megatron tp={mt_tp} pp={mt_pp} cp={mt_cp} ep={mt_ep}, vllm_pp={vllm_pp}"
    )

    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type deepscaler "
        "--num-rollout 2 --rollout-batch-size 4 --n-samples-per-prompt 4 "
        "--rollout-max-response-len 4096 --rollout-temperature 0.8 "
        "--global-batch-size 16 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {mt_tp} --sequence-parallel "
        f"--pipeline-model-parallel-size {mt_pp} "
        f"--context-parallel-size {mt_cp} "
        f"--expert-model-parallel-size {mt_ep} --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.0 --kl-loss-type k1 --kl-coef 0.0 "
        "--entropy-coef 0.0 --eps-clip 0.2 "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    vllm_args = (
        f"--rollout-num-gpus {rollout_gpus} "
        f"--rollout-num-gpus-per-engine {engine_gpus} "
        f"--num-gpus-per-node {num_gpus_per_node} "
        f"--vllm-pipeline-parallel-size {vllm_pp} "
        "--vllm-gpu-memory-utilization 0.6 --vllm-max-num-seqs 256 "
        "--vllm-enable-expert-parallel "
    )

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {actor_nodes} --actor-num-gpus-per-node {actor_gpus_per_node} "
        f"{'--colocate ' if mode == 'colocate' else ''}"
        "--moe-token-dispatcher-type alltoall "
    )

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} {vllm_args} {ci_args} {misc_args} "
    )

    # Ray cluster sees total GPUs on this node (8 per H200 host), not just
    # the actor's share — rollout needs its own GPU pool for disagg configs.
    ray_num_gpus_per_node = 8
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=ray_num_gpus_per_node,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(v, None)
    execute()
