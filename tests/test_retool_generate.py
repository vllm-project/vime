"""CPU unit tests for the ``examples/retool`` vLLM rollout port.

Covers the parts of the example that the slime -> vime port actually changed:
the ``/inference/v1/generate`` request body, the ``choices[0]`` response parse,
the multi-turn tool loop, and the tool-concurrency limit. The engine is mocked,
so no GPU or running router is required.
"""

from __future__ import annotations

import asyncio
import sys
import types
from argparse import Namespace
from pathlib import Path

_tests_root = Path(__file__).resolve().parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest

_unit_stubs.install_rollout_optional_stubs()

if not _unit_stubs.real_module_available("psutil"):
    # tool_sandbox uses psutil only for RSS-based cleanup heuristics.
    _psutil = types.ModuleType("psutil")

    class _FakeProcess:
        def memory_info(self):
            return types.SimpleNamespace(rss=64 * 1024 * 1024)

    _psutil.Process = _FakeProcess
    sys.modules["psutil"] = _psutil

# The RL script puts the example dir on PYTHONPATH so `generate_with_retool`
# resolves as a top-level module and can import its sibling `tool_sandbox`.
_RETOOL_DIR = _tests_root.parent / "examples" / "retool"
if str(_RETOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_RETOOL_DIR))

import generate_with_retool as mod  # noqa: E402
import tool_sandbox  # noqa: E402

from vime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


class _FakeTokenizer:
    """Character-code tokenizer: reversible, so decode(encode(t)) == t."""

    def __call__(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(t)) for t in token_ids)


class _FakeState:
    def __init__(self, args):
        self.tokenizer = _FakeTokenizer()
        self.processor = None


def _args(**overrides) -> Namespace:
    args = Namespace(
        partial_rollout=False,
        hf_checkpoint="/fake/qwen3-4b",
        vllm_router_ip="127.0.0.1",
        vllm_router_port=3250,
        rollout_max_context_len=4096,
        context_parallel_size=1,
        max_tokens_per_gpu=4096,
        vllm_speculative_config=None,
        num_layers=2,
        moe_router_topk=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _sampling_params(**overrides) -> dict:
    sp = {"max_new_tokens": 256, "temperature": 1.0, "top_p": 1.0}
    sp.update(overrides)
    return sp


def _choice(text: str, finish_reason: str = "stop", *, logprobs: bool = True) -> dict:
    """Build a vLLM ``/inference/v1/generate`` choice for `text`."""
    token_ids = [ord(c) for c in text]
    choice: dict = {"token_ids": token_ids, "finish_reason": finish_reason}
    if logprobs:
        choice["logprobs"] = {"content": [{"logprob": -0.5} for _ in token_ids]}
    return choice


def _pending_sample(prompt: str = "2+2?") -> Sample:
    return Sample(prompt=prompt, label="4", status=Sample.Status.PENDING)


def _prompt_len(prompt: str = "2+2?") -> int:
    """Token length of the rendered tool-enabled prompt, per _FakeTokenizer."""
    rendered = mod.format_conversation_with_tools(prompt=prompt, tools=mod.tool_registry.get_tool_specs())
    return len(_FakeTokenizer()(rendered)["input_ids"])


@pytest.fixture(autouse=True)
def _patch_state(monkeypatch):
    monkeypatch.setattr(mod, "GenerateState", _FakeState)


@pytest.fixture(autouse=True)
def _stub_tool_subprocess(monkeypatch):
    """Keep the rollout tests off real `python3` subprocesses.

    They exercise the turn loop, not the sandbox, and spawning a subprocess per
    tool call makes them slow and dependent on the runner's environment.
    `test_real_sandbox_executes_code` covers real execution explicitly.
    """

    async def fake_execute_code(code):
        return "Output:\n4"

    monkeypatch.setattr(tool_sandbox.tool_registry.python_sandbox, "execute_code", fake_execute_code)


def _run_generate(monkeypatch, responses, *, args=None, sample=None, sampling_params=None):
    """Drive mod.generate with a scripted list of engine `choices`, capturing payloads."""
    payloads: list[dict] = []
    queue = list(responses)

    async def fake_post(url, payload, **kwargs):
        payloads.append({"url": url, "payload": payload})
        assert queue, "engine called more times than the test scripted"
        return {"choices": [queue.pop(0)]}

    monkeypatch.setattr(mod, "post", fake_post)
    result = asyncio.run(
        mod.generate(
            args or _args(),
            sample if sample is not None else _pending_sample(),
            sampling_params or _sampling_params(),
        )
    )
    return result, payloads


# --------------------------------------------------------------------------
# response parsing (the ported SGLang -> vLLM surface)
# --------------------------------------------------------------------------


def test_parse_vllm_choice_reads_tokens_and_logprobs():
    tokens, log_probs, meta = mod._parse_vllm_choice(_choice("hi"))
    assert tokens == [ord("h"), ord("i")]
    assert log_probs == [-0.5, -0.5]
    assert meta == {"finish_reason": {"type": "stop"}}


@pytest.mark.parametrize(
    ("engine_finish_reason", "expected_type"),
    [("stop", "stop"), ("length", "length"), ("abort", "abort"), ("cancelled", "abort"), (None, "stop")],
)
def test_parse_vllm_choice_normalizes_finish_reason(engine_finish_reason, expected_type):
    choice = _choice("x", finish_reason=engine_finish_reason)
    _, _, meta = mod._parse_vllm_choice(choice)
    assert meta["finish_reason"] == {"type": expected_type}


def test_parse_vllm_choice_passes_through_nested_finish_reason():
    """Defensive: a dict finish_reason (SGLang shape) is used as-is."""
    _, _, meta = mod._parse_vllm_choice({"token_ids": [1], "finish_reason": {"type": "length"}})
    assert meta["finish_reason"] == {"type": "length"}


def test_parse_vllm_choice_reports_missing_logprobs_instead_of_zero_filling():
    tokens, log_probs, _ = mod._parse_vllm_choice(_choice("hi", logprobs=False))
    assert tokens == [ord("h"), ord("i")]
    # Empty, NOT [0.0, 0.0] -- generate() must abort rather than train on fakes.
    assert log_probs == []


# --------------------------------------------------------------------------
# request body
# --------------------------------------------------------------------------


def test_generate_posts_token_ids_body_to_inference_endpoint(monkeypatch):
    _, payloads = _run_generate(monkeypatch, [_choice("Answer: \\boxed{4}")])

    assert len(payloads) == 1
    assert payloads[0]["url"] == "http://127.0.0.1:3250/inference/v1/generate"
    body = payloads[0]["payload"]
    assert body["model"] == "/fake/qwen3-4b"
    assert isinstance(body["token_ids"], list) and body["token_ids"]
    # SGLang's `input_ids` / `return_logprob` must not survive the port.
    assert "input_ids" not in body
    assert "return_logprob" not in body
    # _build_inference_sampling_params renames max_new_tokens and asks for logprobs.
    assert body["sampling_params"]["max_tokens"] == 256
    assert body["sampling_params"]["logprobs"] == 1
    assert "max_new_tokens" not in body["sampling_params"]


def test_generate_prompt_includes_tool_specs(monkeypatch):
    sample = _pending_sample()
    _, payloads = _run_generate(monkeypatch, [_choice("Answer: \\boxed{4}")], sample=sample)

    prompt_text = _FakeTokenizer().decode(payloads[0]["payload"]["token_ids"])
    assert "# Tools" in prompt_text
    assert "code_interpreter" in prompt_text
    assert sample.payload_has_tools is True
    assert sample.payload_has_system is True


# DAPO-Math-17k and AIME-2024 both store `prompt` as a chat-message list, and
# `_build_messages` passes a list through untouched without --apply-chat-template.
LIST_PROMPT = [{"role": "user", "content": "What is 2+2?"}]
LIST_PROMPT_WITH_SYSTEM = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "What is 2+2?"},
]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("plain string", (None, "plain string")),
        (LIST_PROMPT, (None, "What is 2+2?")),
        (LIST_PROMPT_WITH_SYSTEM, ("You are terse.", "What is 2+2?")),
        ([], (None, "")),
    ],
)
def test_split_prompt_handles_both_dataset_shapes(prompt, expected):
    assert mod.split_prompt(prompt) == expected


def test_generate_accepts_a_chat_message_list_prompt(monkeypatch):
    """The real datasets ship list prompts; rendering must still be single-templated."""
    sample = Sample(prompt=LIST_PROMPT, label="4", status=Sample.Status.PENDING)
    _, payloads = _run_generate(monkeypatch, [_choice("Answer: \\boxed{4}")], sample=sample)

    rendered = _FakeTokenizer().decode(payloads[0]["payload"]["token_ids"])
    assert "What is 2+2?" in rendered
    assert rendered.count("<|im_start|>user") == 1, "list prompt must not double-wrap"
    assert rendered.count("<|im_start|>assistant") == 1
    # Passing the list straight to the template renders its repr, which still
    # *contains* the question -- so assert the repr artifacts are absent instead.
    assert "'role'" not in rendered, f"prompt list was rendered as a repr: {rendered[-200:]}"
    assert "{'" not in rendered


def test_generate_uses_a_system_message_from_the_prompt_list(monkeypatch):
    sample = Sample(prompt=LIST_PROMPT_WITH_SYSTEM, label="4", status=Sample.Status.PENDING)
    _, payloads = _run_generate(monkeypatch, [_choice("Answer: \\boxed{4}")], sample=sample)

    rendered = _FakeTokenizer().decode(payloads[0]["payload"]["token_ids"])
    assert "You are terse." in rendered
    assert rendered.count("<|im_start|>system") == 1


def test_reward_func_accepts_a_chat_message_list_prompt():
    """Regression: `sample.prompt + sample.response` raised TypeError on the AIME
    eval set, crashing RolloutManager.eval() at the first --eval-interval."""
    sample = Sample(prompt=LIST_PROMPT, label="4", status=Sample.Status.COMPLETED)
    sample.response = " Answer: \\boxed{4}"
    result = asyncio.run(mod.reward_func(_args(), sample))
    assert result["score"] > 0


def test_prompt_has_exactly_one_conversation_structure():
    """Guards against the nested-`user` prompt that --apply-chat-template produces."""
    rendered = mod.format_conversation_with_tools(prompt="2+2?", tools=mod.tool_registry.get_tool_specs())

    assert rendered.count("<|im_start|>system") == 1
    assert rendered.count("<|im_start|>user") == 1
    assert rendered.count("<|im_start|>assistant") == 1
    # Jinja strips the template's trailing newline, so the open generation turn is
    # `<|im_start|>assistant` with no "\n" (upstream behaviour, preserved).
    assert rendered.endswith("<|im_start|>assistant")
    assert rendered.count("<|im_end|>") == 2


def test_retool_rl_script_does_not_apply_chat_template():
    script = (_tests_root.parent / "examples" / "retool" / "retool_qwen3_4b_rl.sh").read_text()
    active = [ln for ln in script.splitlines() if ln.strip().startswith("--apply-chat-template")]
    assert not active, "retool renders its own chat template; --apply-chat-template double-wraps the prompt"


def test_generate_clamps_per_turn_budget_to_remaining_context(monkeypatch):
    """A single turn must not be allowed to exceed the remaining context budget."""
    headroom = 48
    args = _args(rollout_max_context_len=_prompt_len() + headroom)
    _, payloads = _run_generate(
        monkeypatch,
        [_choice("Answer: \\boxed{4}")],
        args=args,
        sampling_params=_sampling_params(max_new_tokens=10_000),
    )
    assert payloads[0]["payload"]["sampling_params"]["max_tokens"] == headroom


# --------------------------------------------------------------------------
# turn loop
# --------------------------------------------------------------------------


def test_generate_completes_on_boxed_answer(monkeypatch):
    sample, payloads = _run_generate(monkeypatch, [_choice("Answer: \\boxed{4}")])

    assert len(payloads) == 1, "an answer must end the loop"
    assert sample.status is Sample.Status.COMPLETED
    assert sample.response == "Answer: \\boxed{4}"
    assert sample.tool_call_count == 0
    assert sample.response_length == len(sample.response)
    assert sample.loss_mask == [1] * sample.response_length
    assert len(sample.rollout_log_probs) == sample.response_length


def test_generate_runs_tool_then_answers(monkeypatch):
    """A code turn feeds <interpreter> output back and continues to a second turn."""
    sample, payloads = _run_generate(
        monkeypatch,
        [
            _choice("<code>print(2+2)</code>"),
            _choice("Answer: \\boxed{4}"),
        ],
    )

    assert len(payloads) == 2, "tool turn must trigger a follow-up generation"
    assert sample.status is Sample.Status.COMPLETED
    assert sample.tool_call_count == 1
    assert "<interpreter>" in sample.response
    assert "4" in sample.response.split("<interpreter>")[1]

    # Turn 2 must resend prompt + everything generated/observed so far.
    assert len(payloads[1]["payload"]["token_ids"]) > len(payloads[0]["payload"]["token_ids"])

    # Tool tokens are masked out; model tokens are trainable.
    assert len(sample.loss_mask) == sample.response_length
    assert set(sample.loss_mask) == {0, 1}
    assert len(sample.rollout_log_probs) == sample.response_length


def test_generate_masks_only_the_observation_tokens(monkeypatch):
    sample, _ = _run_generate(
        monkeypatch,
        [_choice("<code>print(2+2)</code>"), _choice("Answer: \\boxed{4}")],
    )
    observation = sample.response[sample.response.index("\n\n<interpreter>") :]
    observation = observation[: observation.index("</interpreter>") + len("</interpreter>") + 2]
    assert sample.loss_mask.count(0) == len(observation), "exactly the tool output is masked"


def test_generate_truncates_on_length_finish_reason(monkeypatch):
    sample, payloads = _run_generate(monkeypatch, [_choice("thinking hard", finish_reason="length")])

    assert len(payloads) == 1, "length stop must end the loop"
    assert sample.status is Sample.Status.TRUNCATED


def test_generate_aborts_on_abort_finish_reason(monkeypatch):
    sample, _ = _run_generate(monkeypatch, [_choice("partial", finish_reason="abort")])

    assert sample.status is Sample.Status.ABORTED
    assert sample.response == "", "aborted sample carries no trainable response"


def test_generate_aborts_when_engine_omits_logprobs(monkeypatch):
    """Must not zero-fill: that would desync rollout_log_probs from the tokens."""
    sample, _ = _run_generate(monkeypatch, [_choice("hello", logprobs=False)])

    assert sample.status is Sample.Status.ABORTED
    assert sample.rollout_log_probs is None


def test_generate_aborts_on_logprob_length_mismatch(monkeypatch):
    bad = _choice("hello")
    bad["logprobs"]["content"] = bad["logprobs"]["content"][:2]  # 5 tokens, 2 logprobs
    sample, _ = _run_generate(monkeypatch, [bad])

    assert sample.status is Sample.Status.ABORTED


def test_generate_stops_at_max_tool_calls(monkeypatch):
    max_calls = tool_sandbox.TOOL_CONFIGS["max_tool_calls"]
    # Always emit code, never an answer: the loop must stop itself.
    sample, payloads = _run_generate(
        monkeypatch,
        [_choice("<code>print(1)</code>") for _ in range(max_calls + 5)],
        args=_args(rollout_max_context_len=200_000),
    )
    assert sample.tool_call_count == max_calls
    assert len(payloads) <= max_calls + 1


def test_generate_resets_stale_state_from_a_retried_sample(monkeypatch):
    """Aborted/partial samples come back with state from the first attempt."""
    sample = _pending_sample()
    sample.response = "stale text"
    sample.response_length = 3
    sample.rollout_log_probs = [-1.0, -1.0, -1.0]
    sample.loss_mask = [1, 1, 1]
    sample.tokens = [1, 2, 3]
    sample.status = Sample.Status.PENDING

    sample, _ = _run_generate(monkeypatch, [_choice("Answer: \\boxed{4}")], sample=sample)

    assert "stale text" not in sample.response
    assert sample.response_length == len(sample.response)
    assert len(sample.rollout_log_probs) == sample.response_length
    assert len(sample.loss_mask) == sample.response_length


def test_generate_truncates_when_observation_overflows_context(monkeypatch):
    """Tool output is unbounded, so it must be trimmed to the context budget."""
    code = "<code>print(2+2)</code>"
    # Leave room for the code turn but not for the whole <interpreter> block.
    args = _args(rollout_max_context_len=_prompt_len() + len(code) + 10)

    sample, _ = _run_generate(monkeypatch, [_choice(code), _choice("Answer: \\boxed{4}")], args=args)

    assert sample.status is Sample.Status.TRUNCATED
    assert len(sample.tokens) <= args.rollout_max_context_len
    # response text is resynced from the trimmed tokens
    assert sample.response_length == len(sample.response)
    assert len(sample.loss_mask) == sample.response_length


def test_generate_rejects_partial_rollout(monkeypatch):
    with pytest.raises(AssertionError):
        _run_generate(monkeypatch, [_choice("x")], args=_args(partial_rollout=True))


# --------------------------------------------------------------------------
# prediction parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "action", "content"),
    [
        ("Answer: \\boxed{4}", "answer", "4"),
        ("Answer: \\boxed{\\frac{1}{2}}", "answer", "\\frac{1}{2}"),
        ("<code>print(1)</code>", "code", "print(1)"),
        ('<tool_call>{"name": "code_interpreter", "arguments": {"code": "print(1)"}}</tool_call>', "code", "print(1)"),
        ("```python\nprint(1)\n```", "code", "print(1)"),
        ("just some prose", None, ""),
    ],
)
def test_postprocess_predictions(text, action, content):
    assert mod.postprocess_predictions(text) == (action, content)


def test_postprocess_predictions_parses_pretty_printed_tool_call():
    """Newlines between JSON tokens must not be escaped -- that is a parse error,
    and the dropped tool call silently degrades into the "invalid action" reprompt."""
    text = '<tool_call>\n{\n  "name": "code_interpreter",\n  "arguments": {"code": "print(1)"}\n}\n</tool_call>'
    assert mod.postprocess_predictions(text) == ("code", "print(1)")


def test_postprocess_predictions_recovers_raw_newlines_inside_code():
    """Raw newlines *inside* the code string are invalid JSON; escaping recovers them."""
    text = '<tool_call>{"name": "code_interpreter", "arguments": {"code": "import math\nprint(math.sqrt(16))"}}</tool_call>'
    action, code = mod.postprocess_predictions(text)
    assert action == "code"
    assert code == "import math\nprint(math.sqrt(16))"


def test_postprocess_predictions_prefers_answer_over_code():
    text = "<code>print(1)</code>\nAnswer: \\boxed{7}"
    assert mod.postprocess_predictions(text) == ("answer", "7")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<code>print(1)</code>trailing junk", "<code>print(1)</code>"),
        ("Answer: \\boxed{4} and then rambling", "Answer: \\boxed{4}"),
        ("```python\nprint(1)\n```junk", "```python\nprint(1)\n```"),
        ("nothing to trim", "nothing to trim"),
    ],
)
def test_postprocess_responses_trims_after_last_complete_tag(text, expected):
    assert mod.postprocess_responses(text) == expected


def test_execute_predictions_invalid_action_reprompts():
    next_obs, done = asyncio.run(mod.execute_predictions("prose with no action"))
    assert done is False
    assert "previous action is invalid" in next_obs


def test_execute_predictions_answer_is_terminal():
    next_obs, done = asyncio.run(mod.execute_predictions("Answer: \\boxed{4}"))
    assert (next_obs, done) == ("", True)


# --------------------------------------------------------------------------
# tool concurrency
# --------------------------------------------------------------------------


class _CountingSemaphore:
    """asyncio.Semaphore that records how many times it was acquired."""

    def __init__(self, value: int):
        self._sem = asyncio.Semaphore(value)
        self.acquires = 0

    async def __aenter__(self):
        self.acquires += 1
        await self._sem.acquire()
        return self

    async def __aexit__(self, *exc_info):
        self._sem.release()


def _install_semaphore(monkeypatch, value: int) -> _CountingSemaphore:
    """Point every reference to the tool semaphore at one counting instance.

    ``generate_with_retool`` may hold its own ``from tool_sandbox import SEMAPHORE``
    alias, so patching only ``tool_sandbox.SEMAPHORE`` would leave a second,
    independent semaphore behind and hide a double-acquire.
    """
    sem = _CountingSemaphore(value)
    monkeypatch.setattr(tool_sandbox, "SEMAPHORE", sem)
    monkeypatch.setattr(mod, "SEMAPHORE", sem, raising=False)
    return sem


def _stub_sandbox(monkeypatch, on_execute=None):
    async def fake_execute_code(code):
        if on_execute is not None:
            await on_execute()
        return "Output:\n4"

    monkeypatch.setattr(tool_sandbox.tool_registry.python_sandbox, "execute_code", fake_execute_code)


def test_execute_predictions_takes_the_tool_semaphore_exactly_once(monkeypatch):
    """Pinned to 1 permit, a double-acquire self-deadlocks."""
    sem = _install_semaphore(monkeypatch, 1)
    _stub_sandbox(monkeypatch)

    async def run():
        return await asyncio.wait_for(mod.execute_predictions("<code>print(2+2)</code>"), timeout=5)

    next_obs, done = asyncio.run(run())
    assert done is False
    assert "<interpreter>" in next_obs and "4" in next_obs
    assert sem.acquires == 1, f"tool semaphore acquired {sem.acquires}x per call, expected 1"


def test_concurrent_tool_calls_reach_the_configured_concurrency(monkeypatch):
    """All `tool_concurrency` calls must be able to run at once.

    Gated on a barrier rather than a sleep: each call blocks inside the critical
    section until `limit` of them are in there together. That makes the peak an
    invariant instead of a scheduling race -- and a double-acquire, which only
    fits limit//2 callers, can never fill the barrier and trips the timeout.
    """
    limit = 4
    sem = _install_semaphore(monkeypatch, limit)

    live = 0
    peak = 0
    barrier = asyncio.Event()

    async def track():
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        if live >= limit:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), timeout=10)
        live -= 1

    _stub_sandbox(monkeypatch, on_execute=track)

    async def run():
        tasks = [mod.execute_predictions("<code>print(2+2)</code>") for _ in range(limit * 3)]
        return await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)

    results = asyncio.run(run())
    assert len(results) == limit * 3
    assert peak == limit, f"expected {limit} concurrent tool executions, saw {peak}"
    assert sem.acquires == limit * 3, f"expected 1 acquire per call, got {sem.acquires} for {limit * 3} calls"


# --------------------------------------------------------------------------
# sandbox
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    ["import os", "eval('1')", "open('/etc/passwd')", "__import__('os')", "import numpy"],
)
def test_sandbox_rejects_unsafe_code(code):
    ok, _ = tool_sandbox.tool_registry.python_sandbox._check_code_safety(code)
    assert ok is False


@pytest.mark.parametrize("code", ["print(2+2)", "import math\nprint(math.sqrt(16))", "x = sum(range(10))"])
def test_sandbox_allows_plain_math(code):
    ok, message = tool_sandbox.tool_registry.python_sandbox._check_code_safety(code)
    assert ok is True, message


def test_real_sandbox_executes_code():
    """The one test that actually spawns the sandbox subprocess.

    A fresh PythonSandbox sidesteps the autouse stub on the registry's instance.
    """
    sandbox = tool_sandbox.PythonSandbox(timeout=60, memory_limit="1GB")
    out = asyncio.run(sandbox.execute_code("print(2 + 2)"))
    assert "4" in out, out


def test_real_sandbox_reports_rejected_code():
    sandbox = tool_sandbox.PythonSandbox(timeout=60, memory_limit="1GB")
    out = asyncio.run(sandbox.execute_code("import os\nprint(os.getcwd())"))
    assert "Error" in out and "not allowed" in out.lower() or "dangerous" in out.lower(), out


def test_unknown_tool_is_reported_not_raised():
    result = asyncio.run(tool_sandbox.tool_registry.execute_tool("nope", {}))
    assert "not found" in result


# --------------------------------------------------------------------------
# reward
# --------------------------------------------------------------------------


def test_reward_func_scores_correct_answer():
    sample = _pending_sample()
    sample.response = " Answer: \\boxed{4}"
    result = asyncio.run(mod.reward_func(_args(), sample))
    assert result["score"] > 0


def test_reward_func_wrong_answer_gets_tool_use_bonus_but_stays_negative():
    sample = _pending_sample()
    sample.response = " Answer: \\boxed{5}"
    sample.tool_call_count = 8
    result = asyncio.run(mod.reward_func(_args(), sample))
    assert result["score"] <= -0.6
    assert result["pred"] is not None


def test_reward_func_rejects_non_sample():
    with pytest.raises(TypeError):
        asyncio.run(mod.reward_func(_args(), {"response": "x"}))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
