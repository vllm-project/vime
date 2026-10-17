"""Compare DSpark loss forward/backward latency and peak CUDA allocation.

Run from the repository root with PYTHONPATH=. See examples/dspark/README.md.
The optional baseline file is loaded as Python code; use a trusted checkout.
"""

import argparse
import gc
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from vime.backends.megatron_utils.dspark.common import DSparkConfig, DSparkForwardOutput
from vime.backends.megatron_utils.dspark.loss import compute_dspark_loss


def load_baseline(path):
    name = "vime.backends.megatron_utils.dspark._benchmark_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_dspark_loss


def clear_grads(outputs):
    outputs.draft_logits.grad = None
    if outputs.confidence_pred is not None:
        outputs.confidence_pred.grad = None


def step(fn, outputs, config):
    clear_grads(outputs)
    loss, _ = fn(outputs=outputs, config=config)
    loss.backward()


def measure(fn, outputs, config, iterations):
    clear_grads(outputs)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iterations):
        step(fn, outputs, config)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    peak = torch.cuda.max_memory_allocated()
    return {"ms": elapsed, "peak_mib": peak / 2**20, "incremental_peak_mib": (peak - initial) / 2**20}


def check_equivalence(functions, outputs, config, dtype):
    """Check both implementations using identical inputs before timing them."""
    expected = None
    for fn in functions.values():
        clear_grads(outputs)
        loss, metrics = fn(outputs=outputs, config=config)
        loss.backward()
        values = [loss.detach().cpu(), torch.tensor(list(metrics.values()))]
        values.append(outputs.draft_logits.grad.detach().cpu().clone())
        if outputs.confidence_pred is not None:
            values.append(outputs.confidence_pred.grad.detach().cpu().clone())
        if expected is not None:
            for actual, reference in zip(values, expected, strict=True):
                torch.testing.assert_close(actual, reference, rtol=2e-5 if dtype == torch.float32 else 1e-2, atol=1e-7)
        expected = values
    clear_grads(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-loss", type=Path)
    parser.add_argument("--anchors", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--modes", nargs="+", choices=["both", "l1", "confidence", "ce"], default=["both"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(*args.anchors, args.vocab_size, args.block_size, args.batch_size, args.iterations, args.repeats) < 1:
        parser.error("shape dimensions, iterations and repeats must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    dtype = getattr(torch, args.dtype)
    functions = {"current": compute_dspark_loss}
    if args.baseline_loss:
        functions = {"baseline": load_baseline(args.baseline_loss), **functions}
    print(
        json.dumps(
            {
                "environment": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(),
                    "config": {
                        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                    },
                }
            }
        ),
        flush=True,
    )
    for anchors in args.anchors:
        for mode in args.modes:
            torch.manual_seed(42)
            shape = (args.batch_size, anchors, args.block_size, args.vocab_size)
            outputs = DSparkForwardOutput(
                draft_logits=torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True),
                aligned_target_logits=torch.randn(shape, device="cuda", dtype=dtype),
                target_ids=torch.randint(args.vocab_size, shape[:-1], device="cuda"),
                eval_mask=torch.rand(shape[:-1], device="cuda") > 0.1,
                block_keep_mask=torch.ones(shape[:2], device="cuda", dtype=torch.bool),
                confidence_pred=(
                    torch.randn(shape[:-1], device="cuda", requires_grad=True)
                    if mode in ("both", "confidence")
                    else None
                ),
            )
            config = DSparkConfig(l1_loss_alpha=0.9 if mode in ("both", "l1") else 0.0)
            check_equivalence(functions, outputs, config, dtype)
            for fn in functions.values():
                for _ in range(args.warmup):
                    step(fn, outputs, config)
            samples = {name: [] for name in functions}
            for repeat in range(args.repeats):
                order = list(functions)
                if repeat % 2:
                    order.reverse()
                for name in order:
                    samples[name].append(measure(functions[name], outputs, config, args.iterations))
            result = {"shape": shape, "dtype": args.dtype, "mode": mode, "samples": samples}
            result["median"] = {
                name: {key: statistics.median(sample[key] for sample in runs) for key in runs[0]}
                for name, runs in samples.items()
            }
            if "baseline" in samples:
                result["speedup"] = result["median"]["baseline"]["ms"] / result["median"]["current"]["ms"]
            print(json.dumps(result), flush=True)
            del outputs
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
