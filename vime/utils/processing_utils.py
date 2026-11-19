import base64
import io
import json
import logging
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    template_path = Path(name_or_path) / "chat_template.json"
    if getattr(tokenizer, "chat_template", None) is None and template_path.is_file():
        with template_path.open() as template_file:
            tokenizer.chat_template = json.load(template_file).get("chat_template")
    return tokenizer


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:

    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    # return_tensors=None for text (input_ids as lists), "pt" for modality-specific outputs
    result["text_kwargs"] = {
        **result.get("text_kwargs", {}),
        "return_tensors": None,
        "return_mm_token_type_ids": False,
    }
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    audio_value = result.get("audio")
    if isinstance(audio_value, list):
        result["audio"] = [item[0] if isinstance(item, tuple) else item for item in audio_value]

    return result


def _try_load_glm4v_processor(name_or_path: str, **kwargs):
    """Fallback: manually construct a Glm4vProcessor for GLM-4.6V / GLM-4.5V models.

    AutoProcessor fails for these models on transformers < 5.0 because
    the Glm46VProcessor / Glm4vMoeProcessor classes are not registered.
    The underlying Glm4vProcessor (non-MoE) works for both variants since
    they share the same vision architecture.
    """
    try:
        from transformers.models.glm4v.image_processing_glm4v import Glm4vImageProcessor
        from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
        from transformers.models.glm4v.video_processing_glm4v import Glm4vVideoProcessor
    except ImportError:
        return None

    pp_path = Path(name_or_path) / "preprocessor_config.json"
    vp_path = Path(name_or_path) / "video_preprocessor_config.json"
    if not pp_path.exists():
        return None

    skip_keys = {"image_processor_type", "processor_class", "video_processor_type"}
    with open(pp_path) as f:
        pp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
    image_processor = Glm4vImageProcessor(**pp_cfg)

    video_processor = None
    if vp_path.exists():
        with open(vp_path) as f:
            vp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
        video_processor = Glm4vVideoProcessor(**vp_cfg)

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    proc = Glm4vProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )
    logger.info(f"Loaded Glm4vProcessor manually for {name_or_path}")
    return proc


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer instead of a proper processor, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        # Fallback: try to construct a GLM-4.6V / GLM-4.5V processor manually.
        proc = _try_load_glm4v_processor(name_or_path, **kwargs)

    return proc


def _extract_images_from_messages(messages):
    """Extract PIL images from chat messages containing multimodal content.

    Handles base64 strings (with or without data: URI prefix), file paths,
    and PIL Image objects embedded in message content dicts.
    """
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            image_data = item.get("image")
            if image_data is None:
                continue
            if isinstance(image_data, Image.Image):
                images.append(image_data)
            elif isinstance(image_data, str):
                if image_data.startswith("data:"):
                    _, encoded = image_data.split(",", 1)
                    images.append(Image.open(io.BytesIO(base64.b64decode(encoded))))
                else:
                    try:
                        raw = base64.b64decode(image_data)
                        images.append(Image.open(io.BytesIO(raw)))
                    except Exception:
                        # Not base64 — try as file path
                        images.append(Image.open(image_data))
    return images


def _load_audio(source):
    import soundfile as sf

    audio, sample_rate = sf.read(source, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(sample_rate), 16000)
        audio = resample_poly(audio, 16000 // divisor, int(sample_rate) // divisor).astype("float32")
        sample_rate = 16000
    return audio, sample_rate


def _extract_audios_from_messages(messages):
    audios = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "audio":
                continue
            audio = item.get("audio")
            if audio is None:
                audio = item.get("audio_url")
            if isinstance(audio, str) and audio.startswith("data:"):
                audio = io.BytesIO(base64.b64decode(audio.split(",", 1)[1]))
            if isinstance(audio, str) and audio.startswith(("http://", "https://", "file://")):
                audios.append(audio)
            elif isinstance(audio, str) or hasattr(audio, "read"):
                audios.append(_load_audio(audio))
            elif isinstance(audio, tuple):
                audios.append(audio)
            elif hasattr(audio, "shape"):
                audios.append((audio, 16000))
    return audios


# Image processors whose own sizing rule must see the original image. For these
# the qwen_vl_utils pre-resize below is skipped, and each image is resized once,
# here, to the size the processor itself picks.
#
# Upstream Slime rationale: the image then reaches two different sizing
# rules -- the HF processor builds the training prompt, SGLang's own processor the
# rollout prompt -- and for Nemotron 3.5 they disagree (HF rounds patches with
# round(x + 0.5), SGLang with round(x)). On geo3k that is 1570 of 2702 images
# with different image-token counts. Qwen's resize happens to hide it by snapping
# every image to a multiple of 32 first. The HF target size is a fixed point of
# both rules, so resizing to it makes both sides build the same grid -- and, since
# neither then resizes again, the same pixels.
_NATIVE_RESIZE_IMAGE_PROCESSORS = frozenset({"NemotronH_Omni_Reasoning_V3ImageProcessor"})


def uses_native_resize(processor) -> bool:
    """Whether process_vision_info resizes to the processor's own target size."""
    image_processor = getattr(processor, "image_processor", None)
    return type(image_processor).__name__ in _NATIVE_RESIZE_IMAGE_PROCESSORS


def _resize_to_native_target(images, image_processor):
    """Resize each image to the size `image_processor` would resize it to.

    The size comes from the processor itself (`imgs_sizes`, one (h, w) per image),
    so this does not restate its sizing rule. One call over all of a prompt's
    images, so the processor's per-prompt budget applies as it would at train time.
    The resize uses that processor's own kernel (torch bicubic, antialias), rounded
    back to uint8 so the same image can be handed to the rollout engine as a PNG.
    """
    import numpy as np
    import torch

    images = [image.convert("RGB") for image in images]
    sizes = image_processor(images=images, return_tensors=None)["imgs_sizes"]
    resized = []
    for image, (height, width) in zip(images, sizes, strict=True):
        if image.size != (width, height):
            pixels = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)[None].float()
            pixels = torch.nn.functional.interpolate(
                pixels, size=(height, width), mode="bicubic", align_corners=False, antialias=True
            )
            image = Image.fromarray(pixels.round().clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).numpy())
        resized.append(image)
    return resized


def process_vision_info(prompt, processor):
    """Extract image, video, and audio inputs from chat messages.

    Processors in _NATIVE_RESIZE_IMAGE_PROCESSORS get their images resized to their
    own target size (see there). Otherwise tries qwen_vl_utils first (Qwen VL
    family), falls back to generic extraction for other models (e.g. GLM-4.6V).
    """
    audios = _extract_audios_from_messages(prompt) or None
    if uses_native_resize(processor):
        images = _extract_images_from_messages(prompt) or None
        if images:
            images = _resize_to_native_target(images, processor.image_processor)
        return {"images": images, "videos": None, "audio": audios}

    try:
        from qwen_vl_utils import process_vision_info as qwen_process_vision_info

        if hasattr(processor.image_processor, "patch_size"):
            image_patch_size = processor.image_processor.patch_size
        else:
            image_patch_size = DEFAULT_PATCH_SIZE
        images, videos = qwen_process_vision_info(prompt, image_patch_size=image_patch_size)
    except Exception:
        # Fallback: generic extraction for non-Qwen models
        images = _extract_images_from_messages(prompt) or None
        videos = None

    return {"images": images, "videos": videos, "audio": audios}


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def encode_audio_for_rollout_engine(audio) -> str:
    if isinstance(audio, str):
        return audio
    if not isinstance(audio, tuple) or len(audio) != 2:
        raise ValueError(f"Unsupported audio type: {type(audio)}; expected tuple or URL str")
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, *audio, format="WAV", subtype="FLOAT")
    return f"data:audio/wav;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def encode_video_for_rollout_engine(video) -> str:
    if isinstance(video, str):
        return video

    import numpy as np
    import torch

    if isinstance(video, torch.Tensor):
        frames = video.detach().cpu().float()
        if frames.dim() == 4 and frames.shape[1] in (1, 3):
            frames = frames.permute(0, 2, 3, 1)  # (N, H, W, C)
        frames = (frames.clamp(0, 1).numpy() * 255).astype(np.uint8)
        video = [Image.fromarray(frame) for frame in frames]

    if isinstance(video, list) and video:
        encoded = []
        for frame in video:
            if not isinstance(frame, Image.Image):
                raise ValueError(f"Unsupported video frame type: {type(frame)}")
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
        return f"data:video/jpeg;base64,{','.join(encoded)}"

    raise ValueError(f"Unsupported video type: {type(video)}")


def build_multimodal_messages(prompt: str, multimodal_inputs: dict | None):
    multimodal_inputs = multimodal_inputs or {}
    content = [{"type": "text", "text": prompt}]
    encoders = {
        "images": ("image_url", encode_image_for_rollout_engine),
        "audio": ("audio_url", encode_audio_for_rollout_engine),
        "videos": ("video_url", encode_video_for_rollout_engine),
    }
    for key, (media_type, encoder) in encoders.items():
        for value in multimodal_inputs.get(key) or []:
            content.append({"type": media_type, media_type: {"url": encoder(value)}})
    return [{"role": "user", "content": content}] if len(content) > 1 else None
