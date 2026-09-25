"""Focused PR86/main-initialization contracts on the pinned Vime Megatron runtime."""

import errno
import gc
import os
from types import SimpleNamespace

import pytest
import torch

from vime_plugins.optimizers import nvme_stream as stream


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("bucket_count", [1, 8])
@requires_cuda
def test_direct_initialization_exact_bytes_releases_storage_and_bounds_peak(tmp_path, bucket_count):
    entries_per_bucket, elements_per_entry = 2, 1024 * 1024
    stager = stream._Stager(256 * 1024)
    buckets = []
    for bucket_id in range(bucket_count):
        entries = []
        for entry_id in range(entries_per_bucket):
            model = torch.nn.Parameter(
                torch.linspace(-1, 1, elements_per_entry, device="cuda", dtype=torch.bfloat16)
                + (bucket_id + entry_id) / 16
            )
            main = torch.empty_like(model, dtype=torch.float32)
            stream._resize(main, 0)
            entries.append(stream._Entry(model, main, 0))
        buckets.append(
            stream._Bucket(
                str(tmp_path / f"bucket{bucket_id}.bin"),
                entries,
                None,
                stager,
                {name: torch.float32 for name in stream.SEGMENTS},
            )
        )
    store = object.__new__(stream.NVMeOptimizerStateStore)
    store.buckets = buckets
    store.dist_opt = SimpleNamespace(
        _get_model_param_range_map=lambda param: {
            "param": SimpleNamespace(start=0, end=param.numel(), size=param.numel())
        }
    )
    identities = [id(entry.main_param) for bucket in buckets for entry in bucket.entries]
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        written = store.initialize_main_from_model_params()
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - baseline
        one_bucket_bytes = entries_per_bucket * elements_per_entry * 4
        assert written == bucket_count * one_bucket_bytes
        # An eight-bucket case must not retain the eight-bucket FP32 main state.
        assert peak_delta <= one_bucket_bytes + 1024 * 1024, peak_delta
        assert torch.cuda.memory_allocated() == baseline
        assert identities == [id(entry.main_param) for bucket in buckets for entry in bucket.entries]
        for bucket in buckets:
            for index, entry in enumerate(bucket.entries):
                assert entry.main_param.untyped_storage().nbytes() == 0
                assert entry.main_param.shape == entry.model_param.shape
                data = os.pread(bucket.fd, elements_per_entry * 4, bucket.offsets["main"][index])
                actual = torch.frombuffer(bytearray(data), dtype=torch.float32)
                torch.testing.assert_close(actual, entry.model_param.detach().float().cpu(), rtol=0, atol=0)
        print(f"NVMe init: buckets={bucket_count} main_bytes={written} peak_delta={peak_delta}")
    finally:
        for bucket in buckets:
            bucket.close()


@pytest.fixture
def megatron_world(tmp_path):
    import torch.distributed as dist
    from megatron.core import parallel_state

    assert not dist.is_initialized(), "Run constructor tests in their own process"
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("deferred", [False, True])
@requires_cuda
def test_real_megatron_constructor_handles_and_peak(megatron_world, deferred):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.transformer import TransformerConfig

    # Multiple equal shards distinguish largest-shard transient allocation from
    # cumulative FP32 main residency. The ordinary path is an explicit control.
    net = torch.nn.Sequential(*[torch.nn.Linear(1024, 1024, bias=False) for _ in range(16)])
    net = net.bfloat16().cuda()
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        net,
    )
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            bf16=True,
            use_distributed_optimizer=True,
            defer_main_param_initialization=deferred,
        ),
        [model],
    )
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline
    optimizer_mains = [param for group in optimizer.param_groups for param in group["params"]]
    model_mains = [param.main_param for param in model.parameters()]
    assert optimizer_mains
    assert {id(param) for param in optimizer_mains} == {id(param) for param in model_mains}
    for param in model_mains:
        assert param.shape == (1024 * 1024,)
        assert param.numel() > 0 and param.is_cuda and param.dtype == torch.float32
        assert param.untyped_storage().nbytes() == (0 if deferred else param.numel() * 4)
    full_bytes = sum(param.numel() * 4 for param in model_mains)
    largest_shard = max(param.numel() * 4 for param in model_mains)
    if deferred:
        assert peak_delta < largest_shard + 2 * 1024 * 1024, peak_delta
    else:
        assert peak_delta >= full_bytes, peak_delta
    print(f"Megatron constructor: deferred={deferred} full_main_bytes={full_bytes} peak_delta={peak_delta}")


def test_short_io_retries_and_eof_fails():
    buffer = bytearray(12)
    offsets = []

    def short_write(fd, buffers, offset):
        offsets.append(offset)
        return min(3, len(buffers[0]))

    stream._rw_full(short_write, 0, 20, buffer)
    assert offsets == [20, 23, 26, 29]
    with pytest.raises(OSError, match="short"):
        stream._rw_full(lambda fd, buffers, offset: 0, 0, 0, buffer)


def test_enospc_is_not_silently_converted_to_sparse_file(tmp_path, monkeypatch):
    path = tmp_path / "full.bin"
    with path.open("wb") as handle:

        def no_space(*args):
            raise OSError(errno.ENOSPC, "injected disk full")

        monkeypatch.setattr(os, "posix_fallocate", no_space)
        with pytest.raises(OSError) as failure:
            stream._reserve(handle.fileno(), 4096)
        assert failure.value.errno == errno.ENOSPC


def test_failed_file_reservation_closes_descriptor(tmp_path, monkeypatch):
    opened = []
    original_open = os.open

    def tracking_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def no_space(*args):
        raise OSError(errno.ENOSPC, "injected disk full")

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(stream, "_reserve", no_space)
    with pytest.raises(OSError):
        stream._allocate_file(str(tmp_path / "full.bin"), 4096)
    try:
        with pytest.raises(OSError) as failure:
            os.fstat(opened[0])
        assert failure.value.errno == errno.EBADF
    finally:
        # Keep the regression safe even when run against the unfixed implementation.
        try:
            os.close(opened[0])
        except OSError:
            pass


@pytest.mark.parametrize("cleanup", ["close", "gc"])
def test_bucket_descriptor_lifetime(tmp_path, cleanup):
    param = torch.zeros(1)
    bucket = stream._Bucket(
        str(tmp_path / "bucket.bin"),
        [stream._Entry(param, param, 0)],
        None,
        None,
        dict.fromkeys(stream.SEGMENTS, torch.float32),
    )
    fd = bucket.fd
    if cleanup == "close":
        bucket.close()
        assert bucket.fd is None
    else:
        del bucket
        gc.collect()
    with pytest.raises(OSError) as failure:
        os.fstat(fd)
    assert failure.value.errno == errno.EBADF
    # Reusing the freed fd must be safe from a second close or later finalization.
    with (tmp_path / "replacement").open("wb") as replacement:
        if cleanup == "close":
            bucket.close()
            del bucket
            gc.collect()
        os.fstat(replacement.fileno())
