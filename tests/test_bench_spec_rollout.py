"""CPU tests for tools/bench_spec_rollout.py (plan, dry run and summary)."""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

NUM_GPUS = 0

_PATH = Path(__file__).resolve().parents[1] / "tools" / "bench_spec_rollout.py"
_SPEC = importlib.util.spec_from_file_location("bench_spec_rollout", _PATH)
bench = importlib.util.module_from_spec(_SPEC)
sys.modules["bench_spec_rollout"] = bench
_SPEC.loader.exec_module(bench)


def _plan(tmp_path, **overrides):
    values = dict(
        out=str(tmp_path),
        num_rollout=5,
        seed=7,
        num_gpus=4,
        modes=list(bench.MODES),
        layouts=list(bench.LAYOUTS),
        env=["MODEL_DIR=/data/Qwen3-4B"],
        extra_args=[],
        example=bench.DEFAULT_EXAMPLE,
    )
    values.update(overrides)
    return bench.plan(Namespace(**values))


@pytest.mark.unit
def test_plan_covers_every_mode_and_layout_with_the_same_seed(tmp_path):
    result = _plan(tmp_path)

    assert [arm["name"] for arm in result["arms"]] == [
        f"{layout}-{mode}" for layout in bench.LAYOUTS for mode in bench.MODES
    ]
    for arm in result["arms"]:
        env = arm["env"]
        assert env["SPEC_MODE"] == arm["mode"]
        # Every arm trains the policy, so acceptance is measured under the same drift.
        assert env["DSPARK_FREEZE_POLICY"] == "0"
        assert env["NUM_ROLLOUT"] == "5" and env["NUM_GPUS"] == "4"
        # No checkpoint, not even the one train.py writes after the last rollout.
        assert env["NO_SAVE"] == "1" and "SAVE_DIR" not in env
        # No aime evaluation before step 0 either.
        assert env["EVAL_INTERVAL"] == "off"
        assert "--seed 7" in env["EXTRA_ARGS"] and "--rollout-seed 7" in env["EXTRA_ARGS"]
        assert env["TENSORBOARD_DIR"] == str(Path(arm["dir"]) / "tensorboard")
        assert env["MODEL_DIR"] == "/data/Qwen3-4B"
        assert Path(arm["script"]).exists()
    assert json.loads((tmp_path / "plan.json").read_text()) == result


@pytest.mark.unit
def test_draft_arms_serve_another_draft_without_dspark_training(tmp_path):
    spec = {"method": "dflash", "model": "/data/Qwen3-4B-speculator.dflash2", "num_speculative_tokens": 8}
    result = _plan(
        tmp_path, modes=["off"], layouts=["colocate"], drafts=[bench.parse_draft("dflash2=" + json.dumps(spec))]
    )

    assert [arm["name"] for arm in result["arms"]] == ["colocate-off", "colocate-dflash2"]
    draft = result["arms"][1]
    assert draft["mode"] == "dflash2"
    # SPEC_MODE=off drops the example's DSpark config and training flags; the draft comes from EXTRA_ARGS.
    assert draft["env"]["SPEC_MODE"] == "off"
    words = draft["env"]["EXTRA_ARGS"].split()
    assert json.loads(words[words.index("--vllm-speculative-config") + 1]) == spec
    assert "--seed 7" in draft["env"]["EXTRA_ARGS"]
    assert "--vllm-speculative-config" not in result["arms"][0]["env"]["EXTRA_ARGS"]


@pytest.mark.unit
def test_cli_plans_only_draft_arms_when_modes_is_empty(tmp_path, monkeypatch):
    spec = '{"method":"dflash","model":"/m/d","num_speculative_tokens":8}'
    argv = ["bench", "plan", "--out", str(tmp_path), "--layouts", "colocate", "--modes", "--drafts", f"dflash2={spec}"]
    monkeypatch.setattr(sys, "argv", argv)
    bench.main()

    arms = json.loads((tmp_path / "plan.json").read_text())["arms"]
    assert [arm["name"] for arm in arms] == ["colocate-dflash2"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        'online={"model":"/m"}',  # clashes with a built-in mode
        'Dflash2={"model":"/m"}',
        'dflash2={"method":"dflash"}',  # no model
        'dflash2={"model":"/my models/d"}',  # EXTRA_ARGS is word-split
    ],
)
def test_parse_draft_rejects_bad_arms(text):
    with pytest.raises(bench.argparse.ArgumentTypeError):
        bench.parse_draft(text)


@pytest.mark.unit
def test_example_option_selects_the_mtp_example(tmp_path):
    result = _plan(tmp_path, layouts=["colocate"], example="examples/mtp/run-qwen3.5-4B-mtp-{layout}.sh")

    for arm in result["arms"]:
        assert arm["script"].endswith("examples/mtp/run-qwen3.5-4B-mtp-colocate.sh")
        assert Path(arm["script"]).exists()
        assert arm["env"]["SPEC_MODE"] == arm["mode"]


@pytest.mark.unit
def test_dry_run_runs_nothing(tmp_path, capsys):
    _plan(tmp_path, layouts=["colocate"])

    code = bench.run(Namespace(out=str(tmp_path), arms=["colocate-off"], dry_run=True))

    assert code == 0
    assert 'SPEC_MODE="off"' in capsys.readouterr().out
    assert not (tmp_path / "colocate-off" / "run.json").exists()


@pytest.mark.unit
def test_arm_without_tensorboard_events_fails(tmp_path):
    result = _plan(tmp_path, layouts=["colocate"], modes=["off"])
    script = tmp_path / "fake.sh"
    script.write_text("exit 0\n")
    result["arms"][0]["script"] = str(script)
    (tmp_path / "plan.json").write_text(json.dumps(result))

    code = bench.run(Namespace(out=str(tmp_path), arms=None, dry_run=False))

    assert code == 1
    record = json.loads((tmp_path / "colocate-off" / "run.json").read_text())
    assert record["exit_code"] == 0 and record["tensorboard_event_files"] == 0


@pytest.mark.unit
def test_summary_skips_warmup_and_splits_acceptance_into_thirds():
    scalars = {
        "perf/rollout_time": {0: 100.0, 1: 10.0, 2: 12.0, 3: 14.0},
        "rollout/spec_accept_rate": {0: 0.9, 1: 0.6, 2: 0.5, 3: 0.4},
        # The token-weighted key wins over the per-sample mean when both exist.
        "rollout/spec_accept_length/token_weighted": {0: 9.0, 1: 3.0, 2: 3.0, 3: 3.0},
        "rollout/spec_accept_length": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
    }

    summary = bench.summarize_arm(scalars, warmup=1)

    assert summary["rollout_time"] == pytest.approx(12.0)
    assert summary["rollout_time_steps"] == 3
    assert summary["spec_accept_rate_first_third"] == pytest.approx(0.6)
    assert summary["spec_accept_rate_last_third"] == pytest.approx(0.4)
    assert summary["spec_accept_length"] == pytest.approx(3.0)
    assert "logprob_abs_diff" not in summary


@pytest.mark.unit
def test_speedup_is_against_off_arm_of_the_same_layout():
    arms = [
        {"name": f"{layout}-{mode}", "layout": layout, "mode": mode}
        for layout in bench.LAYOUTS
        for mode in ("off", "online")
    ]
    summaries = {
        "colocate-off": {"rollout_time": 30.0},
        "colocate-online": {"rollout_time": 20.0},
        "non-colocate-off": {"rollout_time": 40.0},
        "non-colocate-online": {"rollout_time": 10.0},
    }

    rows = {row["arm"]: row for row in bench.compare(summaries, arms)}

    assert rows["colocate-online"]["rollout_time_speedup_vs_off"] == pytest.approx(1.5)
    assert rows["non-colocate-online"]["rollout_time_speedup_vs_off"] == pytest.approx(4.0)
    assert rows["colocate-off"]["rollout_time_speedup_vs_off"] == pytest.approx(1.0)


@pytest.mark.unit
def test_summarize_reads_tensorboard_written_by_several_processes(tmp_path):
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter

    _plan(tmp_path, layouts=["colocate"], modes=["off", "online"])
    for name, rollout_times in (("colocate-off", [50.0, 30.0, 30.0]), ("colocate-online", [50.0, 20.0, 20.0])):
        tb_dir = tmp_path / name / "tensorboard"
        # The rollout manager and the trainer write separate event files.
        rollout_writer, train_writer = SummaryWriter(str(tb_dir)), SummaryWriter(str(tb_dir))
        for step, value in enumerate(rollout_times):
            rollout_writer.add_scalar("perf/rollout_time", value, step)
            train_writer.add_scalar("train/train_rollout_logprob_abs_diff", 0.01, step)
        rollout_writer.close()
        train_writer.close()

    rows = {row["arm"]: row for row in bench.summarize(Namespace(out=str(tmp_path), warmup=1))}

    assert rows["colocate-online"]["rollout_time_speedup_vs_off"] == pytest.approx(1.5)
    assert rows["colocate-online"]["logprob_abs_diff"] == pytest.approx(0.01)
    assert "| colocate-online |" in (tmp_path / "summary.md").read_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
