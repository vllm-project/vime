"""Small real-Megatron/NCCL gradient snapshot smoke test.

Run with torchrun; see docs/en/developer_guide/parallel_gradient_check.md.
"""


def main():
    import argparse
    import hashlib
    import json
    import os
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    from vime.backends.megatron_utils.gradient_check import save_gradient_snapshot
    from vime.backends.megatron_utils.update_weight.common import all_gather_param, named_params_and_buffers

    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--overlap", action="store_true")
    p.add_argument("--per-token-loss", action="store_true")
    p.add_argument("--lr", type=float, default=1e-6)
    a = p.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    mpu.initialize_model_parallel(tensor_model_parallel_size=a.tp, pipeline_model_parallel_size=a.pp)
    model_parallel_cuda_manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        ffn_hidden_size=128,
        tensor_model_parallel_size=a.tp,
        pipeline_model_parallel_size=a.pp,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        gradient_accumulation_fusion=False,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16 if a.bf16 else torch.float32,
        calculate_per_token_loss=a.per_token_loss,
        bf16=a.bf16,
        params_dtype=torch.bfloat16 if a.bf16 else torch.float32,
    )
    net = GPTModel(
        config,
        get_gpt_layer_local_spec(),
        vocab_size=128,
        max_sequence_length=32,
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
    ).cuda()
    if a.bf16:
        from megatron.core.transformer.module import Float16Module

        net = Float16Module(config, net)
    from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes

    for param in net.parameters():
        set_defaults_if_not_set_tensor_model_parallel_attributes(param)
    net = DDP(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=True, grad_reduce_in_fp32=True, overlap_grad_reduce=a.overlap, bucket_size=8192
        ),
        net,
    )
    args = SimpleNamespace(num_experts=None, fp16=False, swiglu=True)
    # Generate each logical tensor independently, then shard it exactly as Megatron does.
    for name, param in named_params_and_buffers(args, [net]):
        shape = list(param.shape)
        sharded = getattr(param, "tensor_model_parallel", False)
        dim = getattr(param, "partition_dim", 0)
        if sharded:
            shape[dim] *= a.tp
        g = torch.Generator().manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
        value = torch.randn(shape, generator=g) * 0.02
        if "norm" in name and param.ndim == 1:
            value += 1
        if sharded and a.tp > 1:
            if "linear_fc1." in name:
                gate, up = value.chunk(2, dim=0)
                value = torch.cat(
                    [
                        gate.chunk(a.tp)[mpu.get_tensor_model_parallel_rank()],
                        up.chunk(a.tp)[mpu.get_tensor_model_parallel_rank()],
                    ]
                )
            else:
                value = value.chunk(a.tp, dim=dim)[mpu.get_tensor_model_parallel_rank()]
        param.data.copy_(value)
    config.finalize_model_grads_func = finalize_model_grads
    if a.overlap:
        config.no_sync_func = net.no_sync
    opt = get_megatron_optimizer(
        OptimizerConfig(
            lr=a.lr, min_lr=a.lr, weight_decay=0.1, use_distributed_optimizer=True, bf16=a.bf16, clip_grad=1.0
        ),
        [net],
    )
    net.zero_grad_buffer()
    opt.zero_grad()
    seq = 16
    samples = 4
    dp_size = mpu.get_data_parallel_world_size()
    dp_rank = mpu.get_data_parallel_rank()
    indices = iter(range(dp_rank, samples, dp_size))
    outputs = []

    def forward(data, model):
        index = next(data)
        tokens = ((torch.arange(seq, device="cuda") + index * 7) % 128).unsqueeze(0)
        targets = (tokens + 1) % 128
        positions = torch.arange(seq, device="cuda").unsqueeze(0)
        mask = torch.triu(torch.ones(seq, seq, device="cuda", dtype=torch.bool), diagonal=1)[None, None]
        losses = model(tokens, positions, mask, labels=targets)

        def loss_fn(losses):
            # Uneven per-sample masks exercise sample averaging and accumulation.
            valid = (torch.arange(seq, device="cuda") < seq - index - 1).float().unsqueeze(0)
            loss = (losses.float() * valid).sum() / valid.sum()
            outputs.append(dict(sample=index, selected_logprobs=(-losses.detach().float()).cpu()))
            if a.per_token_loss:
                return (losses.float() * valid).sum(), valid.sum().int(), {"loss": loss.detach()}
            return loss, {"loss": loss.detach()}

        return losses, loss_fn

    get_forward_backward_func()(
        forward_step_func=forward,
        data_iterator=indices,
        model=[net],
        num_microbatches=samples // dp_size,
        seq_length=seq,
        micro_batch_size=1,
        forward_only=False,
    )
    root = Path(a.out)
    save_gradient_snapshot(args, [net], root / "grads")
    # Snapshot collection must leave every live buffer unchanged.
    before = [b.grad_data.clone() for b in net.buffers]
    save_gradient_snapshot(args, [net], root / "grads-repeat")
    assert all(torch.equal(x, b.grad_data) for x, b in zip(before, net.buffers, strict=True))
    success, norm, zeros = opt.step()
    assert success
    weights = {}
    for name, param in named_params_and_buffers(args, [net]):
        full = all_gather_param(name, param)
        if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
            weights[name] = full.detach().cpu().clone()
    root.mkdir(parents=True, exist_ok=True)
    torch.save(dict(weights=weights, outputs=outputs, norm=float(norm)), root / f"result-{dist.get_rank()}.pt")
    if dist.get_rank() == 0:
        print(
            json.dumps(
                dict(
                    tp=a.tp,
                    pp=a.pp,
                    dp=dp_size,
                    bf16=a.bf16,
                    overlap=a.overlap,
                    grad_norm=float(norm),
                    success=success,
                )
            ),
            flush=True,
        )
    dist.barrier()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
