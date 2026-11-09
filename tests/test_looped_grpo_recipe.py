"""Check actor-only launch and fresh-resume contracts without starting Ray."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("family", ["nanbeige"])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("estimator", ["grpo", "ppo", "dppo", "flow-dppo"])
@pytest.mark.parametrize("group_size", [None, 3])
def test_recipe_roles_groups_precision_and_resume(tmp_path, monkeypatch, family, resume, estimator, group_size):
    source = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("looped_recipe", source / "examples/looped_ppo/run.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    model, output = tmp_path / "model", tmp_path / "run"
    model.mkdir()
    config = {
        "model_type": family,
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "vocab_size": 11,
        "intermediate_size": 32,
        "max_position_embeddings": 256,
        "rope_theta": 10000,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "num_loops": 2,
        "n_layers": 3,
        "n_embd": 16,
        "n_heads": 2,
        "padded_vocab_size": 11,
        "block_size": 256,
        "rope_base": 10000,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
        "mean_recurrence": 32,
    }
    (model / "config.json").write_text(json.dumps(config))
    arguments = [
        "run.py",
        "--model",
        str(model),
        "--data",
        str(tmp_path / "math.jsonl"),
        "--output",
        str(output),
        "--model-revision",
        "checkpoint-pin",
        "--engine-revision",
        "engine-pin",
        "--ray-address",
        "host:6379",
        "--rollout-batch-size",
        "2",
        "--algorithm",
        estimator,
    ]
    if group_size is not None:
        arguments.extend(["--n-samples-per-prompt", str(group_size)])
    if resume:
        checkpoint = output / "checkpoints/actor"
        checkpoint.mkdir(parents=True)
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("1")
        arguments.append("--resume")
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    captured = {}
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(init=lambda **kwargs: captured.update(ray=kwargs)))
    monkeypatch.setattr(recipe.runpy, "run_path", lambda *args, **kwargs: captured.update(command=sys.argv[1:]))
    if estimator == "grpo":
        recipe.main(advantage_estimator="grpo")
    else:
        recipe.main()
    command = captured["command"]

    def flag(name):
        return command[command.index("--" + name) + 1]

    roles = json.loads((output / "roles.json").read_text())["megatron"]
    assert [role["role"] for role in roles] == (["actor"] if estimator == "grpo" else ["actor", "critic"])
    assert flag("advantage-estimator") == ("grpo" if estimator == "grpo" else "ppo")
    count = group_size if group_size is not None else (4 if estimator == "grpo" else 1)
    assert flag("n-samples-per-prompt") == str(count)
    assert flag("global-batch-size") == str(count if estimator == "flow-dppo" else 2 * count)
    assert ("--use-tis" in command) is (estimator == "dppo")
    assert ("--use-rollout-logprobs" in command) is (estimator != "dppo")
    if estimator == "dppo":
        assert flag("tis-clip-low") == "0.5" and flag("tis-clip") == "2.0"
    if estimator == "flow-dppo":
        assert flag("flow-dppo-divergence-budget") == "0.01" and flag("num-steps-per-rollout") == "2"
    assert flag("rlt-start-version") == ("2" if resume else "0")
    assert flag("load") == str(output / "checkpoints/actor" if resume else model)
    assert flag("rlt-model-revision") == "checkpoint-pin" and flag("rlt-engine-revision") == "engine-pin"
    if estimator == "grpo":
        assert "--normalize-advantages" not in command
        assert "--value-clip" not in command and "--lambd" not in command
    else:
        assert "--normalize-advantages" in command
        assert flag("value-clip") == "0.2" and flag("lambd") == "0.95"
    if family == "nanbeige":
        assert "--recurrent-fp32" in command and flag("kv-channels") == "128"
        assert flag("rlt-kv-blocks") == "44"
    else:
        assert "--fp16" in command and flag("loss-scale") == "128"
        assert flag("rlt-kv-blocks") == "704"
    assert captured["ray"]["address"] == "host:6379"
    assert captured["ray"]["runtime_env"]["py_executable"] == sys.executable
    assert not (output / "checkpoints/critic").exists()


def test_grpo_rejects_single_completion_before_starting_ray(monkeypatch):
    from examples.looped_ppo.run import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--model",
            "unused",
            "--data",
            "unused",
            "--output",
            "unused",
            "--model-revision",
            "pin",
            "--engine-revision",
            "pin",
            "--n-samples-per-prompt",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main(advantage_estimator="grpo")
    assert error.value.code == 2
