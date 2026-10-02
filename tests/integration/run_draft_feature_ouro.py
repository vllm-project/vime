"""Run the existing Ouro recipe with VIME's target-forward collector.

First argument: examples/ouro/train.py. Remaining arguments are its usual CLI.
Use --only-train-params-name-list lm_head.weight for a small-memory CUDA check.
This wrapper retains the real RLT generation, reward, trainer and publication.
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch
from megatron.training.global_vars import get_args
from safetensors.torch import save_file

from vime.backends.megatron_utils.draft_feature_collector import DraftFeatureCollector
from vime.backends.megatron_utils.loss import get_log_probs_and_entropy
from vime.backends.megatron_utils.model import forward_only
from vime.utils.draft_feature_contract import manifest_from_dict


class LogitCollector(DraftFeatureCollector):
    def forward(self, model, forward_kwargs, batch, rollout_data, indices):
        before = self.published
        output = super().forward(model, forward_kwargs, batch, rollout_data, indices)
        if self.published != before:
            manifest = manifest_from_dict(json.loads((self.root / f"batch-{before:04d}.json").read_text()))
            positions = torch.tensor(
                [token.packed_position for token in manifest.token_map.selected_tokens], device=output.device
            )
            expected = output.detach().reshape(-1, output.shape[-1]).index_select(0, positions).cpu()
            # Verification tensors are outside the consumer artifact directory.
            verification = self.root.parent.parent / "verification" / self.root.name
            verification.mkdir(parents=True, exist_ok=True)
            save_file({"logits": expected}, verification / f"batch-{before:04d}.safetensors")
        return output


def main():
    recipe = Path(sys.argv.pop(1)).resolve()
    # The official synchronous recipe reuses rollout logprobs. This validation
    # runs the target forward in both off/on cases, as the normal collector path does.
    sys.argv.remove("--use-rollout-logprobs")
    sys.argv[0] = str(recipe)
    spec = importlib.util.spec_from_file_location("ouro_feature_recipe", recipe)
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_train, synchronize = module.train, module.synchronize_plan
    current_plan = []

    def synchronize_plan(plan):
        synchronize(plan)
        current_plan[:] = [plan]

    def train(update, chunks, optimizer, scheduler, iterators, num_microbatches, global_batch_sizes):
        args = get_args()
        data = iterators[0].rollout_data
        size = len(data["tokens"])
        first = update * args.global_batch_size + args.rank * size
        data["sample_indices"] = list(range(first, first + size))
        data["group_indices"] = [(first + index) // args.n_samples_per_prompt for index in range(size)]
        data["rollout_ids"] = list(data["sample_indices"])
        # The recipe checks every actual RLT output against this serving plan.
        data["weight_versions"] = [[str(current_plan[0].policy_version)] for _ in range(size)]
        collector = None
        if args.draft_feature_mode == "collect-only":
            collector = LogitCollector(
                args, module.unwrap_model(chunks)[0], update, str(current_plan[0].policy_version)
            )
        data.update(
            forward_only(
                get_log_probs_and_entropy, args, chunks, iterators, num_microbatches, draft_feature_collector=collector
            )
        )
        if collector is not None:
            collector.finish()
        for iterator in iterators:
            iterator.reset()
        return original_train(update, chunks, optimizer, scheduler, iterators, num_microbatches, global_batch_sizes)

    module.synchronize_plan, module.train = synchronize_plan, train
    module.main()


if __name__ == "__main__":
    main()
