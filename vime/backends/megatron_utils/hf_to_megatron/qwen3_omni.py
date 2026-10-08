from .common import SafetensorReader, strip_mcore_wrappers
from .qwen import qwen_moe_hf_tensor


class _ThinkerReader:
    def __init__(self, reader: SafetensorReader) -> None:
        self.reader = reader

    def __contains__(self, name: str) -> bool:
        return f"thinker.{name}" in self.reader

    def get_tensor(self, name: str):
        return self.reader.get_tensor(f"thinker.{name}")


def qwen3_omni_hf_tensor(name: str, reader: SafetensorReader, config):
    name = strip_mcore_wrappers(name)
    for model_prefix, hf_prefix in (("audio_model.", "audio_tower."), ("vision_model.", "visual.")):
        if name.startswith(model_prefix):
            return reader.get_tensor(f"thinker.{hf_prefix}{name.removeprefix(model_prefix)}")
    return qwen_moe_hf_tensor(name, _ThinkerReader(reader), config.thinker_config.text_config)
