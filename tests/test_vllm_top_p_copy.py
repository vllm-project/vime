"""Sampling metadata must survive overlapped copies and the next forward."""

import pytest
import torch

NUM_GPUS = 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("with_logprobs", [False, True])
def test_top_p_copy_preserves_values_after_source_release(with_logprobs):
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors

    forward = torch.cuda.Stream()
    copy = torch.cuda.Stream()
    for _ in range(32):
        with torch.cuda.stream(forward):
            masks = SamplingMaskTensors(
                token_ids=torch.full((16, 128), 17, dtype=torch.int32, device="cuda"),
                packed_mask=torch.empty((16, 0), dtype=torch.uint8, device="cuda"),
                counts=torch.full((16,), 128, dtype=torch.int32, device="cuda"),
                vocab_size=128,
                logprobs=torch.full((16, 128), -0.5, device="cuda") if with_logprobs else None,
            )
            sampler = SamplerOutput(
                sampled_token_ids=torch.zeros((16, 1), dtype=torch.int32, device="cuda"),
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=torch.ones(16, dtype=torch.int32, device="cuda"),
                num_rejected=torch.zeros(16, dtype=torch.int32, device="cuda"),
                sampling_mask_tensors=masks,
            )
        copy.wait_stream(forward)
        with torch.cuda.stream(copy):
            torch.cuda._sleep(3_000_000)
        result = AsyncOutput(
            ModelRunnerOutput(req_ids=[str(index) for index in range(16)], req_id_to_index={}),
            sampler,
            sampler.num_sampled,
            main_stream=forward,
            copy_stream=copy,
            check_ep_fault=False,
            pending_aux_output=None,
        )
        del masks, sampler
        with torch.cuda.stream(forward):
            next_forward = [torch.full((16, 128), 99, dtype=torch.int32, device="cuda") for _ in range(32)]
        try:
            copied = result.get_output().sampling_masks
            assert copied is not None
            ids = torch.from_numpy(copied.token_ids)
            torch.testing.assert_close(ids, torch.full_like(ids, 17), rtol=0, atol=0)
            if with_logprobs:
                assert copied.logprobs is not None
                scores = torch.from_numpy(copied.logprobs)
                torch.testing.assert_close(scores, torch.full_like(scores, -0.5), rtol=0, atol=0)
            else:
                assert copied.logprobs is None
        finally:
            forward.synchronize()
            del next_forward


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
