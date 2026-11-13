"""Stream complete HF weights through the real Bridge mappings.

The sparse exporter keeps a standalone Bridge iterator; ordinary full and
disk transfers retain the upstream direct iterator and weight source.
"""

import sys
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

from megatron.core.utils import unwrap_model

from .megatron_delta_export import trim_hf_vocab_padding


def _bridge_config_view(config):
    """Detach metadata from Adaptor classes replaced in their defining module.

    Bridge broadcasts a cleaned config across PP ranks. Pickle cannot resolve
    Adaptor-generated/replaced classes even after runtime leaves are removed.
    Preserve their stored metadata in an importable attribute container; live
    training configs are restored when the export context exits.
    """
    cls = type(config)
    resolved = sys.modules.get(cls.__module__)
    for part in cls.__qualname__.split("."):
        resolved = getattr(resolved, part, None)
    if resolved is cls:
        return config
    attributes = {}
    if is_dataclass(config):
        attributes.update((f.name, getattr(config, f.name)) for f in fields(config) if hasattr(config, f.name))
    attributes.update(vars(config))
    return SimpleNamespace(**attributes)


def configure_bridge_module_types():
    """Teach older Bridge releases about Megatron's column-parallel CE head."""
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    from megatron.core.tensor_parallel import ColumnParallelLinear

    try:
        from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule
    except ImportError:
        return
    if not issubclass(LinearCrossEntropyModule, ColumnParallelLinear):
        raise TypeError("Bridge CE-head registration requires ColumnParallelLinear inheritance")
    AutoMapping.register_module_type(LinearCrossEntropyModule.__name__, "column")


class HfWeightIteratorBridge:
    def __init__(self, args, model):
        from megatron.bridge import AutoBridge

        configure_bridge_module_types()
        self.args = args
        self.model = model
        self.bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        # Transformers 5 keeps rope_theta inside rope_parameters.
        config = self.bridge.hf_pretrained.config
        for current in (config, getattr(config, "text_config", None)):
            if current is None:
                continue
            rope = getattr(current, "rope_parameters", None) or getattr(current, "rope_scaling", None)
            if isinstance(rope, dict) and "rope_theta" in rope and not hasattr(current, "rope_theta"):
                current.rope_theta = rope["rope_theta"]
        text_config = getattr(config, "text_config", config)
        self.hf_vocab_size = getattr(text_config, "vocab_size", None)

    @contextmanager
    def model_context(self):
        models = unwrap_model(self.model)
        added = []
        replaced, views = [], {}
        try:
            for model in models:
                for module in model.modules():
                    config = getattr(module, "config", None)
                    if config is None:
                        continue
                    if id(config) not in views:
                        views[id(config)] = _bridge_config_view(config)
                    view = views[id(config)]
                    if view is not config:
                        replaced.append((module, config))
                        module.config = view
                config = model.config
                if not hasattr(config, "share_embeddings_and_output_weights"):
                    config.share_embeddings_and_output_weights = model.share_embeddings_and_output_weights
                    added.append(config)
            yield
        finally:
            for config in added:
                del config.share_embeddings_and_output_weights
            for module, config in reversed(replaced):
                module.config = config

    def get_hf_weight_chunks(self):
        with self.model_context():
            chunk, nbytes = [], 0
            for exported in self.bridge.export_hf_weights(self.model, cpu=False, show_progress=False):
                # Bridge revisions expose either two or three tuple fields.
                name, weight = exported[:2]
                weight = trim_hf_vocab_padding(name, weight, self.hf_vocab_size)
                size = weight.numel() * weight.element_size()
                if chunk and nbytes + size > self.args.update_weight_buffer_size:
                    yield chunk
                    chunk, nbytes = [], 0
                chunk.append((name, weight))
                nbytes += size
            if chunk:
                yield chunk
