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
    "You are a concise math tutor. Given a math Question, the model's Final Answer, "
    "and raw Reasoning Traces, produce a short, faithful explanation of how to solve it. "
    "Use the traces only to ensure correctness and include key ideas; "
    "do NOT quote or replicate the traces verbatim; "
    "avoid step-by-step derivations, hidden thoughts, or any XML tags."
)

SUMMARY_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Final Answer:\n{answer}\n\n"
    "Reasoning Traces (raw & possibly truncated):\n{traces}\n\n"
    "Now provide a concise explanation (about 2–6 sentences) that justifies the final answer "
    "and highlights the main steps and insights. Do not reproduce the traces verbatim."
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

def join_traces(traces: List[str]) -> str:
    """将多段自然推理轨迹拼接为单个字符串，用可视化分隔符隔开。"""
    if not traces:
        return ""
    # 用明显分隔便于模型识别不同片段
    return "\n----- TRACE -----\n".join(traces)

def format_traces_for_prompt(traces: List[str], max_chars: int = 6000) -> str:
    """拼接并按字符数上限截断，避免超长提示。"""
    joined = join_traces(traces)
    if not joined:
        return "(none)"
    if len(joined) <= max_chars:
        return joined
    # 保留前后两段，提示被截断
    head_keep = int(max_chars * 0.7)
    tail_keep = int(max_chars * 0.2)
    head = joined[:head_keep]
    tail = joined[-tail_keep:]
    return f"{head}\n... [TRUNCATED {len(joined) - head_keep - tail_keep} CHARS] ...\n{tail}"

def build_chat_messages(question: str, answer: str, traces_text: str) -> List[Dict[str, str]]:
    """把 Question + Final Answer + Traces 一起放进消息格式。"""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": SUMMARY_USER_TEMPLATE.format(
            question=question, answer=answer, traces=traces_text
        )},
    ]

def build_chat_prompt_via_tokenizer(tokenizer, question: str, answer: str, traces_text: str) -> str:
    messages = build_chat_messages(question, answer, traces_text)
    # HF 官方建议：用 chat_template 把 messages 转字符串
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
    ap.add_argument("--out", default="/root/autodl-fs/out2000/A+Q+RT/math/out_openr1_answer_summaries.jsonl")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    # vLLM 关键参数
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--use_vllm_chat", action="store_true",
                    help="直接传 messages 给 vLLM（支持新版 vLLM，推荐）。")
    ap.add_argument("--offline", action="store_true",
                    help="设置 HF 离线环境变量")
    ap.add_argument("--traces_max_chars", type=int, default=6000,
                    help="拼入提示的推理轨迹最大字符数（过长将智能截断）")
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

    # 将原始 traces 格式化后注入提示
    traces_for_prompt = [
        format_traces_for_prompt(t, max_chars=args.traces_max_chars) for t in traces_list
    ]

    # 准备 messages
    inputs_messages = [
        build_chat_messages(q, a, t) for q, a, t in zip(problems, answers, traces_for_prompt)
    ]
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
        sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        summaries: List[str] = []
        if args.use_vllm_chat:
            # 新版 vLLM：直接传 messages，会自动应用 chat template
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
            inputs_str = [
                build_chat_prompt_via_tokenizer(tokenizer, q, a, t)
                for q, a, t in zip(problems, answers, traces_for_prompt)
            ]
            for batch in chunked(inputs_str, args.batch_size):
                outs = llm.generate(batch, sampling)
                for o in outs:
                    text = o.outputs[0].text if o.outputs else ""
                    summaries.append(text.strip())
    else:
        summaries = [""] * len(problems)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for uid, q, a, s, traces, tprompt in zip(
            uuids, problems, answers, summaries, traces_list, traces_for_prompt
        ):
            rec = {
                "uuid": uid,
                "question": q,
                "final_answer": a,
                # 兼容下游字段名：虽然名字还是 answer_driven，但实际已使用了 traces 做增强
                "summary_answer_driven": s,
                "traces_natural": traces,            # 原始自然推理轨迹（不参与生成）
                "traces_used_in_prompt": tprompt,    # 实际拼入提示（可能被截断）
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
                    "traces_max_chars": args.traces_max_chars,
                }
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(summaries)} records to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
