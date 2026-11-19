"""Native Nemotron image expansion; requires local model config/tokenizer files."""

import os
from pathlib import Path

import pytest
from PIL import Image

NUM_GPUS = 0
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def processor():
    model = Path(os.environ.get("VIME_TEST_NEMOTRON_VL_CKPT", "/root/models/NVIDIA-Nemotron-3.5-Super-EA-09112026"))
    if not (model / "config.json").is_file():
        pytest.skip("Set VIME_TEST_NEMOTRON_VL_CKPT to local model files")
    pytest.importorskip("vllm")
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    return MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(
            model=str(model),
            tokenizer=str(model),
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 2},
            mm_processor_cache_gb=0,
        )
    )


@pytest.fixture
def images():
    return [Image.new("RGB", (320, 240), "red"), Image.new("RGB", (240, 480), "blue")]


def expand(processor, prompt, images):
    token_ids = processor.info.get_tokenizer().encode(prompt, **processor.info.default_tok_params.get_encode_kwargs())
    output = processor(
        token_ids,
        mm_items=processor.info.parse_mm_data({"image": images} if images else {}),
        hf_processor_mm_kwargs={},
    )
    return token_ids, output


@pytest.mark.parametrize("count", [1, 2])
def test_images_expand_once_at_corresponding_placeholders(processor, images, count):
    prompt = " then ".join(["<image>"] * count)
    token_ids, output = expand(processor, prompt, images[:count])
    ranges = output["mm_placeholders"]["image"]
    counts = output["mm_kwargs"].get_data()["num_tokens_per_image"]
    assert len(ranges) == len(counts) == count
    marker = processor.info.get_tokenizer().get_vocab()["<image>"]
    expected = []
    image_index = 0
    for token in token_ids:
        if token != marker:
            expected.append(token)
            continue
        replacement = processor.info.get_hf_processor().get_image_repl(int(counts[image_index]), 1)
        span = ranges[image_index]
        assert marker in replacement.full
        assert span.offset == len(expected)
        assert span.length == len(replacement.full)
        assert span.get_num_embeds() == int(counts[image_index])
        expected.extend(replacement.full)
        image_index += 1
    assert image_index == count
    assert output["prompt_token_ids"] == expected


def test_missing_image_placeholder_is_rejected(processor, images):
    with pytest.raises(AssertionError, match="Failed to apply prompt replacement"):
        expand(processor, "text", images[:1])


def test_extra_image_placeholder_is_rejected_by_training_processor(processor, images):
    from vime.rollout.vllm_rollout import _prepare_prompt_ids
    from vime.utils.processing_utils import load_processor
    from vime.utils.types import Sample

    model = os.environ.get("VIME_TEST_NEMOTRON_VL_CKPT", "/root/models/NVIDIA-Nemotron-3.5-Super-EA-09112026")
    hf_processor = load_processor(model, trust_remote_code=True, local_files_only=True)
    sample = Sample(prompt="<image><image>", multimodal_inputs={"images": images[:1]})
    with pytest.raises(IndexError):
        _prepare_prompt_ids(sample, processor.info.get_tokenizer(), hf_processor)


def test_text_without_images_is_unchanged(processor):
    token_ids, output = expand(processor, "text", [])
    assert output["prompt_token_ids"] == token_ids
    assert not output["mm_placeholders"].get("image")
