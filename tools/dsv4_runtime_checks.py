"""Check the native Megatron DeepSeek-V4 spec before a full GRPO run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    ratios = config["compress_ratios"][: config["num_hidden_layers"]]
    assert ratios[:4] == [0, 0, 4, 128]
    assert set(ratios) == {0, 4, 128}
    from megatron.core.models.hybrid.layers.utils import Symbols

    assert Symbols.DSV4_COMPRESS_RATIO_MAP == {"C": 4, "H": 128, "W": 0}
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention_module_specs import (
        get_dsv4_hybrid_module_spec_for_backend,
    )
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
        DSv4HybridSelfAttention,
    )

    assert callable(get_dsv4_hybrid_module_spec_for_backend)
    assert DSv4HybridSelfAttention.__name__ == "DSv4HybridSelfAttention"
    from vime_plugins.models.deepseek_v4 import get_dsv4_spec

    assert callable(get_dsv4_spec)
    print(f"Megatron native DSv4 available; four-layer ratios={ratios[:4]}")


if __name__ == "__main__":
    main()
