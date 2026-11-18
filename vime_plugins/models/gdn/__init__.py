"""Qwen GDN implementation package.

- ``fla_npu``: AscendC kernel adapters over the fla_npu package. Imports
  ``torch_npu`` and is therefore loaded lazily, only when the ``fla_npu``
  backend is chosen.
- ``qwen3_5_tp``: the opt-in fused TP-sharded Qwen3.5 GatedDeltaNet module
  (``--qwen-gdn-implementation fused_tp``).

``vime_plugins.models.qwen_gdn_backend`` is the shared kernel dispatcher for
every GDN module (replicated path, fused-TP path and Qwen3-Next); it is
imported from here rather than duplicated.
"""
