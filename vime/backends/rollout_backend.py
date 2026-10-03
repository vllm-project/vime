"""Select the rollout deployment without importing optional engine packages."""


def start_rollout_servers(args, pg):
    if args.rollout_backend == "vllm-rlt":
        from vime.backends.vllm_rlt_utils.deployment import start_rollout_servers as start
    else:
        from vime.backends.vllm_utils.deployment import start_rollout_servers as start
    return start(args, pg)
