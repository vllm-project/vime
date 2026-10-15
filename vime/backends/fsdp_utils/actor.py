"""Ray training actor for the FSDP2 backend (``--train-backend fsdp2``).

Same role as ``MegatronTrainRayActor``: ``RayTrainGroup`` and the rollout
manager drive it through ``init``, ``train``, ``save_model``,
``update_weights``, ``sleep``/``wake_up``, ``clear_memory`` and
``set_rollout_manager``. The model is an HF transformers causal LM sharded with
FSDP2 over every training rank; the reference model shares the same sharded
module and is swapped in from host copies of the shards.

The overall flow is adapted from slime's ``FSDPTrainRayActor``
(https://github.com/THUDM/slime, ``slime/backends/fsdp_utils/actor.py`` at
0104a9e9, Apache-2.0), rebuilt on vime's rollout schedule, loss functions and
weight-transfer stack.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from vime.observability import train_data_utils, train_metric_utils
from vime.observability.logging_utils import init_tracking
from vime.observability.profile_utils import TrainProfiler
from vime.observability.timer import Timer, inverse_timer, timer, with_defer
from vime.ray.train_actor import TrainRayActor
from vime.utils import accelerator
from vime.utils.data import process_rollout_data
from vime.utils.distributed_utils import get_gloo_group
from vime.utils.memory_utils import clear_memory, print_memory
from vime.utils.misc import Box

from . import checkpoint
from .arguments import validate_fsdp_args
from .lr_scheduler import get_param_scheduler
from .model import (
    build_fsdp_model,
    build_optimizer,
    initial_weights_path,
    is_hf_checkpoint,
    load_hf_weights_into,
    move_optimizer_state,
)
from .trainer import FSDPTrainer, move_rollout_data_to_device
from .update_weight import FSDPWeightUpdater

logger = logging.getLogger(__name__)


class FSDPTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0
        validate_fsdp_args(args, role)
        if with_opd_teacher:
            raise NotImplementedError("The fsdp2 backend has no Megatron OPD teacher; use --opd-type vllm.")

        super().init(args, role, with_ref, with_opd_teacher)
        # The loss/advantage/metric helpers shared with the Megatron backend read
        # their process groups from megatron.core.mpu; register this trainer's
        # layout there: pure data parallelism over every rank.
        from megatron.core import mpu

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            distributed_timeout_minutes=args.distributed_timeout_minutes,
            create_gloo_process_groups=True,
        )
        self.device = torch.device(accelerator.device_type(), accelerator.current_device())
        self.train_parallel_config = {
            "dp_size": dist.get_world_size(),
            "cp_size": 1,
            "vpp_size": 1,
            "microbatch_group_size_per_vp_stage": 1,
        }
        torch.manual_seed(args.seed)

        if dist.get_rank() == 0:
            init_tracking(args, primary=False, role=role)
        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())
        validate_fsdp_args(args, role, self.hf_config)

        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(accelerator.device_type(), (dist.get_world_size(),), mesh_dim_names=("dp",))
        model = build_fsdp_model(args, self.hf_config, initial_weights_path(args), mesh, self.device)
        optimizer = build_optimizer(args, model)
        self.trainer = FSDPTrainer(args, model, optimizer, get_param_scheduler(args, optimizer), self.device)
        loaded_rollout_id = checkpoint.load(self.trainer)
        start_rollout_id = (loaded_rollout_id if loaded_rollout_id is not None else 0) + 1

        if with_ref:
            self.trainer.backup.backup("actor")
            self.load_other_checkpoint("ref", args.ref_load)
            self.trainer.switch_model("actor")

        if self.args.vocab_size is None:
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        if not hasattr(args, "update_weight_start_version"):
            args.update_weight_start_version = (
                args.start_rollout_id if args.start_rollout_id is not None else start_rollout_id
            )
        self.weight_updater = FSDPWeightUpdater(args, model, self.device)

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from vime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        clear_memory()
        if self.args.offload_train:
            self.sleep()
        self.prof.on_init_end()
        return start_rollout_id

    @timer
    def sleep(self) -> None:
        """Move parameter shards and optimizer states to the host.

        Process groups stay alive; with ``--fsdp-cpu-offload`` the shards already
        live on the host and only the cache is released.
        """
        assert self.args.offload_train
        print_memory("before offload model")
        if not self.args.fsdp_cpu_offload:
            self.trainer.model.to("cpu")
            move_optimizer_state(self.trainer.optimizer, "cpu")
        clear_memory()
        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")
        if not self.args.fsdp_cpu_offload:
            self.trainer.model.to(self.device)
            move_optimizer_state(self.trainer.optimizer, self.device)
        clear_memory()
        print_memory("after wake_up model")

    def compute_log_prob(self, rollout_data: dict, store_prefix: str = "", use_rollout_top_p_replay: bool = True):
        return self.trainer.compute_log_prob(rollout_data, store_prefix, use_rollout_top_p_replay)

    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.debug_rollout_only:
            return None
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = process_rollout_data(rollout_data_ref, dist.get_rank(), dist.get_world_size())
            rollout_data = move_rollout_data_to_device(rollout_data, self.device)

        with inverse_timer("train_wait"), timer("train"):
            self.trainer.prepare_actor_inputs(rollout_data)
            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)
            train_metric_utils.log_rollout_data(rollout_id, self.args, rollout_data)
            with timer("actor_train"):
                self.trainer.train(rollout_id, rollout_data)
            self.prof.step(rollout_id=rollout_id)

        train_data_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)
        train_metric_utils.log_perf_data(rollout_id, self.args)

        if self.args.offload_train:
            del rollout_data
            self.sleep()
        return None

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return
        if self.args.offload_train:
            self.wake_up()
        checkpoint.save(self.trainer, rollout_id)
        if self.args.save_hf is not None and self.role == "actor":
            checkpoint.save_hf(self.trainer, Path(self.args.save_hf.format(rollout_id=rollout_id)), self.tokenizer)
        if self.args.offload_train:
            self.sleep()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if not self.args.rollout_external or self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        (
            rollout_engines,
            rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            engine_parallel_configs,
        ) = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())
        if not rollout_engines:
            if dist.get_rank() == 0:
                logger.info("No updatable vLLM engines are running; skip weight update.")
            return

        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
                engine_parallel_configs=engine_parallel_configs,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        print_memory("before update_weights")
        if self.args.offload_train:
            # With --offload-train the train group preloads torch_memory_saver with tracking enabled, so every CUDA
            # allocation is a VMM mapping that CUDA IPC cannot export ("invalid argument" when sharing the packed
            # buffer). Allocate the transfer buffers untracked, as the Megatron actor does.
            from torch_memory_saver import torch_memory_saver

            with torch_memory_saver.disable():
                self.weight_updater.update_weights()
        else:
            self.weight_updater.update_weights()
        clear_memory()
        print_memory("after update_weights")

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        """Load an HF checkpoint into the sharded module and keep it as ``model_tag``."""
        if not is_hf_checkpoint(path):
            raise ValueError(f"The fsdp2 backend loads {model_tag} weights from an HF directory; got {path!r}.")
        load_hf_weights_into(
            self.trainer.model, path, self.args.fsdp_attn_implementation, cpu_offload=self.args.fsdp_cpu_offload
        )
        self.trainer.backup.backup(model_tag)
