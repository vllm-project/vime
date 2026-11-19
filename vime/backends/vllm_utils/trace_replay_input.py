"""vLLM endpoint plugin for fixed-token replay with incremental prompt input."""

import json
from contextlib import aclosing
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from vllm import SamplingParams
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind


async def _records(request: Request, limit: int):
    pending = b""
    received = 0
    async for data in request.stream():
        received += len(data)
        if received > limit:
            raise ValueError("Trace replay request body exceeds the model's input limit")
        *lines, pending = (pending + data).split(b"\n")
        for line in lines:
            yield json.loads(line)
    if pending:
        yield json.loads(pending)


async def _generate(request: Request):
    engine = request.app.state.trace_replay_engine
    if engine is None or not engine.model_config.enable_trace_replay:
        raise HTTPException(400, "The engine must be started with trace replay enabled")
    if request.headers.get("content-type", "").split(";")[0] != "application/x-ndjson":
        raise HTTPException(415, "Expected application/x-ndjson")
    limit = engine.model_config.max_model_len
    try:
        async with aclosing(_records(request, limit * 64 + 8192)) as records:
            return await _replay(engine, records, limit)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc


async def _replay(engine, records, limit):
    header = await anext(records, None)
    if not isinstance(header, dict):
        raise ValueError("Expected a trace replay header")
    chunk_count = header.get("chunk_count")
    if type(chunk_count) is not int or not 1 <= chunk_count < limit:
        raise ValueError("Invalid prompt chunk count")
    params = SamplingParams(**header["sampling_params"])
    trace = params.trace_decode_token_ids
    if not trace or params.max_tokens != len(trace) or params.n != 1 or params.logprobs != 1:
        raise ValueError("Expected one fixed decode sequence, with one logprob per token")
    params.output_kind = RequestOutputKind.DELTA
    params.stop = []
    interim = params.clone()
    interim.trace_decode_token_ids = trace[:1]
    interim.max_tokens = 1
    prompt_ids = []

    async def chunks():
        for i in range(chunk_count):
            part = await anext(records, None)
            if not isinstance(part, list) or not part or any(type(t) is not int or t < 0 for t in part):
                raise ValueError("Each prompt chunk must contain non-negative integer token IDs")
            prompt_ids.extend(part)
            if len(prompt_ids) + len(trace) > limit:
                raise ValueError("Prompt chunks and decode trace exceed the model context length")
            last = i == chunk_count - 1
            if last and await anext(records, ()) != ():
                raise ValueError("Received more prompt chunks than declared")
            # Earlier chunks use a one-token budget. vLLM emits that token, then
            # discards it when extending the session with the next input chunk.
            yield StreamingInput(
                prompt=TokensPrompt(prompt_token_ids=part), sampling_params=params if last else interim
            )

    request_id = "vime-replay-" + uuid4().hex
    engine.check_admission(request_id=request_id)
    content = []
    finish_reason = None
    engine_prompt_ids = None
    async with aclosing(engine.generate(chunks(), params, request_id)) as output:
        async for update in output:
            if update.prompt_token_ids is not None:
                engine_prompt_ids = update.prompt_token_ids
            for choice in update.outputs:
                for token, scores in zip(choice.token_ids, choice.logprobs or [], strict=True):
                    content.append({"token_id": token, "logprob": scores[token].logprob})
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
    if (
        engine_prompt_ids != prompt_ids
        or len(content) != chunk_count - 1 + len(trace)
        or any(row["token_id"] != trace[0] for row in content[: chunk_count - 1])
    ):
        raise HTTPException(502, "Engine returned an incomplete streaming-input replay")
    content = content[chunk_count - 1 :]
    return {
        "choices": [
            {
                "token_ids": [row["token_id"] for row in content],
                "logprobs": {"content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(content)},
    }


class TraceReplayInputPlugin:
    name = "vime_trace_replay"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        app.add_api_route("/inference/v1/trace-replay-input", _generate, methods=["POST"])

    async def init_state(self, engine_client, state, args) -> None:
        state.trace_replay_engine = engine_client
