"""Run actor-only fixed-depth recurrent GRPO through VIME's standard Ray entry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.looped_ppo.run import main

if __name__ == "__main__":
    main(advantage_estimator="grpo")
