# Local coding-agent smoke test

Runs ten SWE-bench Verified SymPy tasks with one local Docker sandbox per task.
The tested configuration is Qwen3.5-27B-GPTQ-Int4, tensor parallel 8, a 256K
context window, and a 16K response limit. This is rollout-only; it does not train.

## Requirements

- One 8x H200 host with Docker and `vllm/vime:latest`.
- At least 35 GB free while downloading the model.
- This Vime checkout at `/mnt/data/vime-agent-smoke/vime`.
- `sympy-10.jsonl`, Node 22, and the Claude Code npm tarball under the paths below.

## Prepare

```bash
ROOT=/mnt/data/vime-agent-smoke
mkdir -p "$ROOT"/{assets,models,runs,tasks}

cp agent_run/local_multi_turn_smoke/sympy-10.jsonl "$ROOT/tasks/"
curl -L https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz \
  -o "$ROOT/assets/node-v22.20.0-linux-x64.tar.xz"
docker run --rm -v "$ROOT/assets:/out" -w /out node:22 \
  npm pack @anthropic-ai/claude-code
mv "$ROOT"/assets/anthropic-ai-claude-code-*.tgz \
  "$ROOT/assets/anthropic-ai-claude-code.tgz"

docker build -t vime-swe-sympy-23950:local \
  -f agent_run/local_multi_turn_smoke/Dockerfile.sympy-23950 .
docker run --rm -v "$ROOT/models:/models" vllm/vime:latest \
  hf download Qwen/Qwen3.5-27B-GPTQ-Int4 \
  --local-dir /models/Qwen3.5-27B-GPTQ-Int4
```

The ten task IDs are `23950`, `22714`, `22914`, `23534`, `24213`, `23824`,
`23262`, `24066`, `24539`, and `23413`, all prefixed by `sympy__sympy-`.

## Run

```bash
bash agent_run/local_multi_turn_smoke/run_h200.sh
```

Results are written to `$ROOT/runs/latest`: `run.log` contains the aggregate
result, while `trace/<instance_id>/` contains the input, trajectory, source
patch, and grading result for each task.

The official evaluator can produce false positives. In the recorded run,
`sympy__sympy-22714` passed its supplied test but incorrectly accepted
`Point(1 + I, 2)`, so successful rewards still require patch review.
