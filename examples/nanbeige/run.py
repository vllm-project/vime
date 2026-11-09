"""Train Nanbeige with PPO, GRPO, Decoupled PPO, or categorical Flow-DPPO."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.looped_ppo.run import main

if __name__ == "__main__":
    main(model_family="nanbeige")
