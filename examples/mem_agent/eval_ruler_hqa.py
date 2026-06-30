"""
eval_ruler_hqa.py — MemAgent HotpotQA evaluation (vime / vLLM version)
======================================================================
Core logic aligned with slime-agentic eval_ruler_hqa.py and MemAgent ruler_hqa.py.
Inference uses vLLM OpenAI-compatible ``/v1/chat/completions``.

Evaluation data: MemAgent-format eval_{length}.json
    Fields: input, answers, context, num_docs

Metrics: F1, EM, sub_EM

Usage (vLLM server must be running, e.g. via examples/mem_agent/run-eval.sh):
    python eval_ruler_hqa.py \\
        --model  /path/to/hf_ckpt \\
        --tokenizer /path/to/hf_ckpt \\
        --length 200 \\
        --data-root /data/hotpotqa \\
        --save-dir results/ruler_hqa_200 \\
        --save-file iter_0000200

Environment variables:
    VLLM_SERVE_HOST / SERVE_HOST   vLLM server host (default: 127.0.0.1)
    VLLM_SERVE_PORT / SERVE_PORT   vLLM server port (default: 8000)
    MEM_CHUNK_TOKENS               tokens per chunk (default: 2048)
    MEM_MAX_MEMORY                 max tokens for memory update (default: 1024)
    MEM_MAX_FINAL                  max tokens for final answer (default: 256)
    MEM_MAX_CHUNKS                 max number of chunks (default: 512)
    DATAROOT                       directory with eval_*.json
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import aiohttp
from tqdm import tqdm
from transformers import AutoTokenizer

SERVE_HOST = os.getenv("VLLM_SERVE_HOST", os.getenv("SERVE_HOST", "127.0.0.1"))
SERVE_PORT = os.getenv("VLLM_SERVE_PORT", os.getenv("SERVE_PORT", "8000"))
CHUNK_TOKENS = int(os.getenv("MEM_CHUNK_TOKENS", "2048"))
MAX_MEMORY_TOKS = int(os.getenv("MEM_MAX_MEMORY", "1024"))
MAX_FINAL_TOKS = int(os.getenv("MEM_MAX_FINAL", "256"))
MAX_CHUNKS = int(os.getenv("MEM_MAX_CHUNKS", "512"))
MAX_CTX_TOKENS = int(os.getenv("MEM_MAX_CTX_TOKENS", str(10**12)))
BASE_URL = f"http://{SERVE_HOST}:{SERVE_PORT}/v1"
API_KEY = os.getenv("SERVE_API_KEY", "EMPTY")
DATAROOT = os.getenv("DATAROOT", "/data/hotpotqa")

_MEMORY_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

_FINAL_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

_NO_MEMORY = "No previous memory"
_STOP_TOKEN_STRINGS = ["<|im_end|>", "<|endoftext|>"]


def _strip_stop_tokens(text: str) -> str:
    for tok in _STOP_TOKEN_STRINGS:
        text = text.replace(tok, "")
    return text.strip()


def _last_boxed_only_string(s: str) -> str | None:
    if "\\boxed " in s:
        return "\\boxed " + s.split("\\boxed ")[-1].split("$")[0]
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i, right_brace_idx, opens = idx, None, 0
    while i < len(s):
        if s[i] == "{":
            opens += 1
        if s[i] == "}":
            opens -= 1
            if opens == 0:
                right_brace_idx = i
                break
        i += 1
    return s[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def _remove_boxed(s: str) -> str:
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    left = "\\boxed{"
    assert s.startswith(left) and s.endswith("}")
    return s[len(left) : -1]


def _extract_boxed(text: str) -> str:
    s = _last_boxed_only_string(text)
    if s is None:
        return ""
    try:
        return _remove_boxed(s).strip()
    except Exception:
        return ""


def _normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    p = _normalize_answer(prediction).split()
    g = _normalize_answer(ground_truth).split()
    if not p or not g:
        return 0.0, 0.0, 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    prec = num_same / len(p)
    recall = num_same / len(g)
    f1 = 2 * prec * recall / (prec + recall)
    return f1, prec, recall


def _exact_match(prediction: str, ground_truth: str) -> float:
    p = _normalize_answer(prediction)
    g = _normalize_answer(ground_truth)
    if p in ("yes", "no", "noanswer") and p != g:
        return 0.0
    if g in ("yes", "no", "noanswer") and p != g:
        return 0.0
    return float(p == g)


def _sub_exact_match(prediction: str, ground_truth: str) -> float:
    p = _normalize_answer(prediction)
    g = _normalize_answer(ground_truth)
    return float((g in p) or (p in g))


def _agg_metrics(records: list[dict]) -> dict:
    keys = ("judge_f1", "judge_em", "judge_sub_em")
    totals = {k: 0.0 for k in keys}
    n = len(records)
    for r in records:
        for k in keys:
            totals[k] += r[k]
    return {
        "f1": round(totals["judge_f1"] / n, 4) if n else 0.0,
        "em": round(totals["judge_em"] / n, 4) if n else 0.0,
        "sub_em": round(totals["judge_sub_em"] / n, 4) if n else 0.0,
        "total": n,
    }


async def _chat_once(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    payload = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    async with session.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
        data = await resp.json()
    return data["choices"][0]["message"]["content"]


async def _recurrent_infer(
    item: dict,
    model: str,
    tokenizer,
    temperature: float,
    top_p: float,
    sem: asyncio.Semaphore,
) -> str:
    question = item["input"].strip()
    context = item["context"].strip()

    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(ctx_ids) > MAX_CTX_TOKENS:
        half = MAX_CTX_TOKENS // 2
        ctx_ids = ctx_ids[:half] + ctx_ids[-half:]

    chunks = [ctx_ids[i : i + CHUNK_TOKENS] for i in range(0, len(ctx_ids), CHUNK_TOKENS)][:MAX_CHUNKS]

    async with sem:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400)) as session:
            memory = _NO_MEMORY
            for chunk_ids in chunks:
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                msg = _MEMORY_TEMPLATE.format(prompt=question, memory=memory, chunk=chunk_text)
                raw = await _chat_once(
                    session,
                    model,
                    [{"role": "user", "content": msg}],
                    temperature,
                    top_p,
                    MAX_MEMORY_TOKS,
                )
                memory = _strip_stop_tokens(raw) or memory

            final_msg = _FINAL_TEMPLATE.format(prompt=question, memory=memory)
            response = await _chat_once(
                session,
                model,
                [{"role": "user", "content": final_msg}],
                temperature,
                top_p,
                MAX_FINAL_TOKS,
            )
    return response.strip()


async def _openai_infer(
    item: dict,
    model: str,
    tokenizer,
    temperature: float,
    top_p: float,
    max_input_len: int,
    max_output_len: int,
    sem: asyncio.Semaphore,
) -> str:
    question = item["input"].strip()
    context = item["context"].strip()
    prompt = f"{context}\n\n" f"Question: {question}\n" "Please put the answer in \\boxed{}.\n\n" "Answer:"
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_input_len:
        prompt = tokenizer.decode(ids[:max_input_len], skip_special_tokens=True)

    async with sem:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400)) as session:
            response = await _chat_once(
                session,
                model,
                [{"role": "user", "content": prompt}],
                temperature,
                top_p,
                max_output_len,
            )
    return response.strip()


async def run_eval(data: list[dict], args: argparse.Namespace, tokenizer) -> None:
    out_path = Path(args.save_dir) / f"{args.save_file}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cached_ids: set[int] = set()
    if out_path.exists() and not args.force:
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    cached_ids.add(json.loads(line)["_id"])
                except Exception:
                    pass

    todo = [item for item in data if item["_id"] not in cached_ids]
    print(f"Data: {len(data)}  Cached: {len(cached_ids)}  " f"Todo: {len(todo)}  Concurrency: {args.n_proc}")
    if not todo:
        print("All done (cached).")
        _print_existing_stats(out_path)
        return

    sem = asyncio.Semaphore(args.n_proc)

    async def process_one(item: dict) -> dict | None:
        try:
            if args.api == "recurrent":
                response = await _recurrent_infer(item, args.model, tokenizer, args.temperature, args.top_p, sem)
            else:
                response = await _openai_infer(
                    item,
                    args.model,
                    tokenizer,
                    args.temperature,
                    args.top_p,
                    args.max_input_len,
                    args.max_output_len,
                    sem,
                )
        except Exception:
            import traceback

            traceback.print_exc()
            return None

        gold = item["answers"][0] if item.get("answers") else ""
        pred = _extract_boxed(response[-300:]) or ""

        result = {
            "_id": item["_id"],
            "answer": gold,
            "pred": pred,
            "judge_f1": _f1_score(pred, gold)[0],
            "judge_em": _exact_match(pred, gold),
            "judge_sub_em": _sub_exact_match(pred, gold),
            "response": response,
        }
        for k, v in item.items():
            if k not in ("context", "response") and k not in result:
                result[k] = v
        return result

    tasks = [process_one(item) for item in todo]
    records: list[dict] = []
    n_err = 0
    first_shown = False
    fout = open(out_path, "a", encoding="utf-8")
    pbar = tqdm(total=len(tasks), desc=f"ruler_hqa[{args.length}]", dynamic_ncols=True)
    PRINT_INTERVAL = 10

    for coro in asyncio.as_completed(tasks):
        result = await coro
        pbar.update(1)
        if result is None:
            n_err += 1
            pbar.set_postfix(err=n_err, done=len(records))
            continue

        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        fout.flush()
        records.append(result)

        n = len(records)
        rolling_f1 = sum(r["judge_f1"] for r in records) / n
        rolling_sub_em = sum(r["judge_sub_em"] for r in records) / n
        pbar.set_postfix(
            done=n,
            err=n_err,
            f1=f"{rolling_f1 * 100:.1f}",
            sub_em=f"{rolling_sub_em * 100:.1f}",
        )

        if not first_shown:
            first_shown = True
            _print_sample(result)

        if n % PRINT_INTERVAL == 0:
            tqdm.write(
                f"[interim {n}/{len(tasks)}]  "
                f"F1={rolling_f1 * 100:.2f}  "
                f"sub_EM={rolling_sub_em * 100:.2f}  "
                f"err={n_err}"
            )

    pbar.close()
    fout.close()

    all_records = records[:]
    if cached_ids:
        with open(out_path, encoding="utf-8") as f:
            all_records = [json.loads(line) for line in f if line.strip()]

    stats = _agg_metrics(all_records)
    print(f"\n=== ruler_hqa [n_docs={args.length}]  " f"total={stats['total']}  errors={n_err} ===")
    for k in ("f1", "em", "sub_em"):
        print(f"  {k}: {round(stats[k] * 100, 2)}")


def _print_sample(result: dict) -> None:
    sep = "=" * 40
    print(f"\n{sep} Sample {result['_id']} {sep}")
    print(f"[response]  {result['response'][:500]}")
    print(f"[pred]      {result['pred']}")
    print(f"[answer]    {result['answer']}")
    print(f"[sub_em]    {result['judge_sub_em']}")
    print(sep)


def _print_existing_stats(out_path: Path) -> None:
    records = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    if not records:
        return
    stats = _agg_metrics(records)
    print(f"Existing results ({stats['total']} samples):")
    for k in ("f1", "em", "sub_em"):
        print(f"  {k}: {round(stats[k] * 100, 2)}")


def load_data(data_root: str, length: int) -> list[dict]:
    candidates = [
        Path(data_root) / f"eval_{length}.json",
        Path(data_root) / f"eval_{length}.jsonl",
    ]
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            f"Cannot find eval data for length={length} in {data_root!r}. " f"Tried: {[str(p) for p in candidates]}"
        )

    suffix = data_path.suffix.lower()
    if suffix == ".json":
        with open(data_path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = list(raw.values())
    else:
        with open(data_path, encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]

    data = []
    for idx, item in enumerate(raw):
        if "input" in item:
            item = dict(item)
            item.setdefault("_id", idx)
            data.append(item)
        elif "prompt" in item:
            meta = item.get("metadata") or {}
            data.append(
                {
                    "_id": idx,
                    "input": item["prompt"],
                    "answers": meta.get("ground_truth", [item.get("label", "")]),
                    "context": meta.get("context", ""),
                    "num_docs": meta.get("num_docs", 0),
                }
            )
        else:
            print(f"[warn] skipping row {idx}: unrecognized format (keys={list(item.keys())[:5]})")

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="MemAgent HotpotQA eval — vime/vLLM version")
    parser.add_argument(
        "--length",
        type=int,
        default=200,
        choices=[50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600],
        help="Number of distractive documents (controls context length)",
    )
    parser.add_argument(
        "--data-root",
        default=DATAROOT,
        help="Directory containing eval_{length}.json files",
    )
    parser.add_argument("--save-dir", "-s", default="results/ruler_hqa", help="Output directory")
    parser.add_argument("--save-file", "-f", default="model", help="Output filename stem")
    parser.add_argument("--model", "-m", required=True, help="Model id registered in vLLM server")
    parser.add_argument("--tokenizer", "-t", required=True, help="HuggingFace tokenizer path (for chunking)")
    parser.add_argument(
        "--api",
        default="recurrent",
        choices=["recurrent", "openai"],
        help="recurrent: chunk-by-chunk memory update (default); openai: single-turn baseline",
    )
    parser.add_argument("--n-proc", "-n", type=int, default=32, help="Max concurrent requests")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-input-len", type=int, default=120000)
    parser.add_argument("--max-output-len", type=int, default=10000)
    parser.add_argument("--sampling", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-evaluate")
    args = parser.parse_args()

    print(f"[config] vLLM={SERVE_HOST}:{SERVE_PORT}  api={args.api}")
    print(
        f"[config] CHUNK_TOKENS={CHUNK_TOKENS}  MAX_MEMORY={MAX_MEMORY_TOKS}  "
        f"MAX_FINAL={MAX_FINAL_TOKS}  MAX_CHUNKS={MAX_CHUNKS}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    data = load_data(args.data_root, args.length)

    if args.sampling > 1:
        base = data[:]
        data = []
        for s in range(args.sampling):
            for item in base:
                new_item = copy.deepcopy(item)
                new_item["_id"] = item["_id"] * args.sampling + s
                data.append(new_item)

    print(f"[data] {len(data)} samples  (n_docs={args.length})")
    asyncio.run(run_eval(data, args, tokenizer))


if __name__ == "__main__":
    main()
