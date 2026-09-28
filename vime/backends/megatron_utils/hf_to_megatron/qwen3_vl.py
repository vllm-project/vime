from .common import strip_mcore_wrappers
from .qwen import qwen_hf_tensor


class _LanguageReader:
    def __init__(self, reader):
        self.reader = reader

    @staticmethod
    def _key(name):
        return name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name

    def __contains__(self, name):
        return self._key(name) in self.reader

    def get_tensor(self, name):
        return self.reader.get_tensor(self._key(name))


def qwen3_vl_hf_tensor(name, reader, config):
    name = strip_mcore_wrappers(name)
    if name.startswith("model.visual."):
        return reader.get_tensor(name)
    if name == "output_layer.weight" and getattr(config, "tie_word_embeddings", False):
        return reader.get_tensor("model.language_model.embed_tokens.weight")
    return qwen_hf_tensor(name, _LanguageReader(reader), config.text_config)
