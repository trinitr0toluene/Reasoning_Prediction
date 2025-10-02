#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, json, re, sys, os
from typing import List, Dict, Any
from datasets import load_dataset

# vLLM (optional import)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

from transformers import AutoTokenizer

# ===================== Helpers =====================

SUMMARY_SYSTEM_PROMPT = (
    "You are a concise math tutor. Given a math Question and its Final Answer, "
    "write a short, faithful explanation of how one would solve it. "
    "Do NOT show full step-by-step derivations. "
)
SUMMARY_USER_TEMPLATE = (
    "Question:\n{question}\n\nFinal Answer:\n{answer}\n\n"
    "Now provide a concise explanation."
)

THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)

def extract_traces_from_text(txt: str) -> List[str]:
    if not isinstance(txt, str) or not txt:
        return []
    return [m.strip() for m in THINK_TAG_RE.findall(txt)]

def collect_traces(row: Dict[str, Any]) -> List[str]:
    traces = []
    gens = row.get("generations")
    if isinstance(gens, list):
        for g in gens:
            traces.extend(extract_traces_from_text(g))
    sol = row.get("solution")
    traces.extend(extract_traces_from_text(sol if isinstance(sol, str) else ""))
    uniq = []
    for t in traces:
        if t not in uniq:
            uniq.append(t)
    return uniq

def build_chat_messages(question: str, answer: str) -> List[Dict[str, str]]:
    """✅ 这是之前缺失导致 NameError 的函数。"""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": SUMMARY_USER_TEMPLATE.format(question=question, answer=answer)},
    ]

def build_chat_prompt_via_tokenizer(tokenizer, question: str, answer: str) -> str:
    messages = build_chat_messages(question, answer)
    # HF 官方建议的做法：用 chat_template 把 messages 转字符串。:contentReference[oaicite:1]{index=1}
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def chunked(it, n):
    for i in range(0, len(it), n):
        yield it[i:i+n]

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="default", choices=["default", "extended", "all"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--engine", default="vllm", choices=["vllm", "dummy"])
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
                    help="HF repo 或本地目录路径（离线用本地目录）")
    ap.add_argument("--out", default="/root/autodl-fs/out2000/new_prompt/math/out_openr1_answer_summaries.jsonl")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    # vLLM 关键参数
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--use_vllm_chat", action="store_true",
                    help="直接传 messages 给 vLLM（支持新版 vLLM，推荐）。:contentReference[oaicite:2]{index=2}")
    ap.add_argument("--offline", action="store_true",
                    help="设置 HF 离线环境变量")
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"Loading dataset open-r1/OpenR1-Math-220k, subset={args.split} ...", file=sys.stderr)
    # ds = load_dataset("open-r1/OpenR1-Math-220k", args.split, split="train")
    ds = load_dataset("/root/autodl-fs/datasets/OpenR1-Math-220k", "default", split="train")
    if args.limit is not None and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    problems, answers, uuids, traces_list = [], [], [], []
    for row in ds:
        q = row.get("problem") or ""
        a = row.get("answer") or ""
        uid = row.get("uuid") or ""
        problems.append(q)
        answers.append(a)
        uuids.append(uid)
        traces_list.append(collect_traces(row))

    inputs_messages = [build_chat_messages(q, a) for q, a in zip(problems, answers)]
    tokenizer = None

    if args.engine == "vllm":
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not installed. `pip install vllm` or use --engine dummy.")

        llm_kwargs = dict(model=args.model, trust_remote_code=True)
        for k, v in dict(dtype=args.dtype,
                         gpu_memory_utilization=args.gpu_mem_util,
                         max_model_len=args.max_model_len).items():
            if v is not None:
                llm_kwargs[k] = v

        # 兼容不同 vLLM 版本：遇到不识别的 kw 自动剔除
        def safe_build_llm(kwargs):
            while True:
                try:
                    return LLM(**kwargs)
                except TypeError as e:
                    msg = str(e)
                    removed = False
                    for k in list(kwargs.keys()):
                        if k in msg and k != "model":
                            kwargs.pop(k, None)
                            removed = True
                            break
                    if not removed:
                        raise

        print(f"Initializing vLLM with {llm_kwargs}", file=sys.stderr)
        llm = safe_build_llm(llm_kwargs)
        sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)  # :contentReference[oaicite:3]{index=3}

        summaries: List[str] = []
        if args.use_vllm_chat:
            # 新版 vLLM：直接传 messages，会自动应用 chat template。:contentReference[oaicite:4]{index=4}
            try:
                for batch in chunked(inputs_messages, args.batch_size):
                    outs = llm.generate(batch, sampling)
                    for o in outs:
                        text = o.outputs[0].text if o.outputs else ""
                        summaries.append(text.strip())
            except Exception as e:
                print(f"[WARN] messages path failed ({e}); falling back to apply_chat_template.", file=sys.stderr)
                args.use_vllm_chat = False

        if not args.use_vllm_chat:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            inputs_str = [build_chat_prompt_via_tokenizer(tokenizer, q, a)
                          for q, a in zip(problems, answers)]
            for batch in chunked(inputs_str, args.batch_size):
                outs = llm.generate(batch, sampling)
                for o in outs:
                    text = o.outputs[0].text if o.outputs else ""
                    summaries.append(text.strip())
    else:
        summaries = [""] * len(problems)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for uid, q, a, s, traces in zip(uuids, problems, answers, summaries, traces_list):
            rec = {
                "uuid": uid,
                "question": q,
                "final_answer": a,
                "summary_answer_driven": s,   # 仅基于 Q + A 生成
                "traces_natural": traces,     # 原始自然推理轨迹（不参与生成）
                "meta": {
                    "engine": args.engine,
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_tokens,
                    "dtype": args.dtype,
                    "gpu_memory_utilization": args.gpu_mem_util,
                    "max_model_len": args.max_model_len,
                    "use_vllm_chat": bool(args.use_vllm_chat),
                    "offline": bool(args.offline),
                }
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(summaries)} records to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
