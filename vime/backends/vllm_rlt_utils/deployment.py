"""Place a native engine in the standard rollout placement group."""

from dataclasses import dataclass

import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vime.backends.vllm_rlt_utils.engine import NativeEngine
from vime.ray.utils import add_default_ray_env_vars


@dataclass
class NativeServer:
    engines: list[ActorHandle]
    engine_gpu_offsets: list[int]
    engine_gpu_counts: tuple[int, ...] = (1,)
    engine_parallel_configs: tuple[dict[str, int], ...] = ({"tp_size": 1, "pp_size": 1, "pcp_size": 1, "dp_size": 1},)
    update_weights: bool = True
    num_new_engines: int = 1
    router_ip: str | None = None
    prometheus_port: int | None = None
    server_groups: tuple[()] = ()

    @property
    def all_engines(self):
        return self.engines


def start_rollout_servers(args, pg):
    placement, bundles, _ = pg
    offset = 0 if args.debug_rollout_only else args.actor_num_nodes * args.actor_num_gpus_per_node
    engine = (
        ray.remote(NativeEngine)
        .options(
            num_cpus=1,
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement,
                placement_group_bundle_index=bundles[0],
            ),
            runtime_env={"env_vars": add_default_ray_env_vars()},
        )
        .remote()
    )
    args.rlt_engine = engine
    return {"default": NativeServer([engine], [offset])}, [engine.init.remote(args)]
