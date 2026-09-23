"""Stream the same real Bridge mappings used by the sparse delta probe.

Ascend-branch note: the name is ``HfWeightIteratorSparseBridge`` (not
``HfWeightIteratorBridge``) because this branch already ships a
``HfWeightIteratorBridge`` for the ``HfWeightIteratorBase`` registry; the
sparse delta exporter uses this standalone Bridge streaming iterator instead.
"""

from contextlib import contextmanager

from megatron.core.utils import unwrap_model

from .megatron_delta_export import trim_hf_vocab_padding


class HfWeightIteratorSparseBridge:
    def __init__(self, args, model):
        from megatron.bridge import AutoBridge

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
        for model in models:
            config = model.config
            if not hasattr(config, "share_embeddings_and_output_weights"):
                config.share_embeddings_and_output_weights = model.share_embeddings_and_output_weights
                added.append(config)
        try:
            yield
        finally:
            for config in added:
                del config.share_embeddings_and_output_weights

    def get_hf_weight_chunks(self, _weights=None, progress_desc="Bridge weight export"):
        del progress_desc
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
