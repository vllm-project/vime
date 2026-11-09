# Native vLLM-RLT rollout backend

`--rollout-backend vllm-rlt` selects a native Ray-owned engine instead of an HTTP
server. Use `vime.rollout.vllm_rlt_rollout.generate_rollout` as the rollout
function. The ordinary sample, reward, dataset and training contracts remain in
use; backend imports stay lazy for other rollout engines.

The initial implementation supports Ouro and Nanbeige at the checkpoint's
fixed full depth, one dedicated rollout GPU, full-vocabulary sampling and full
physical-weight publication through disk. Configure training GPUs separately.
Colocation, rollout offload/recovery, external HTTP, evaluation and truncated
sampling require additional backend contracts and are rejected by validation.

Install the pinned engine in VIME's isolated environment:

```bash
python -m pip install "git+https://github.com/0z5a/vllm-rlt.git@d2a358933393f25d74dd2bdd1068a741cd5d9226"
```

Supply the same commit with `--rlt-engine-revision`.

Pin the checkpoint and installed RLT source, then supply their identities with
`--rlt-model-revision` and `--rlt-engine-revision`. These flags record provenance;
they do not verify a checkout or download.

Publication pauses generation, checks every safetensors shard, and calls RLT's
public start/update/finish weight transaction. Generation resumes only after
all physical parameters commit. Repeating a committed version requires the
same content digest. Every sample carries the committed policy
version, digest, runtime epoch, model/engine identity and recurrence trace.

A fresh resume supplies `--rlt-start-version` before publishing the restored
actor. Standard training checks the engine's committed version after every
publication. Owned engines close and owned actors exit normally when training
finishes; the existing Ray cluster remains available.

Defaults use BF16 training dtype, Triton attention and eager execution.
`--rlt-attention-backend torch` selects the Torch reference path;
`--rlt-cuda-graphs` opts into graphs. Set cache capacity with `--rlt-kv-blocks`
and admission concurrency with `--rlt-max-num-seqs`.
