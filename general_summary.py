#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, json, os, re, sys
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_dataset

# vLLM (optional)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

from transformers import AutoTokenizer

# ===================== Config / Presets =====================

# 采用问答+轨迹的要点式摘要预设
PRESET_SYSTEM = (
    "You are a concise assistant. Given a Question, its Final Answer, and raw Reasoning Traces, "
    "produce a short, faithful explanation as 3–5 bullet points. "
    "Use the traces only to ensure correctness and capture key insights; "
    "do NOT quote or replicate the traces verbatim, and avoid step-by-step derivations."
)

PRESET_USER_WITH_TRACES = (
    "Question:\n{question}\n\n"
    "Final Answer:\n{answer}\n\n"
    "Reasoning Traces (raw; may be truncated):\n{traces}\n\n"
    "Now provide 3–5 bullet points highlighting the main ideas and a brief verification."
)

# 提取 <think>…</think>
THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
# 常见 “Final Answer:” 标记（英文/中文）
FINAL_ANS_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:final\s*answer|答案)\s*[:：]?\s*(.+)$",
               re.IGNORECASE | re.DOTALL),
]

# ===================== Utilities =====================

def strip_think(resp: str) -> str:
    return THINK_TAG_RE.sub("", resp or "").strip()

def extract_final_answer(resp: str) -> str:
    vis = strip_think(resp)
    for pat in FINAL_ANS_PATTERNS:
        m = pat.search(vis)
        if m:
            return m.group(1).strip()
    lines = [ln.strip() for ln in vis.splitlines() if ln.strip()]
    return lines[-1] if lines else vis

def extract_traces_from_response(resp: str) -> List[str]:
    if not isinstance(resp, str) or not resp:
        return []
    return [m.strip() for m in THINK_TAG_RE.findall(resp)]

def join_traces(traces: List[str]) -> str:
    if not traces:
        return ""
    return "\n----- TRACE -----\n".join(traces)

def format_traces_for_prompt(traces: List[str], max_chars: int = 6000) -> str:
    """拼接并限长；为空时返回空串（避免无谓 token 开销）。"""
    joined = join_traces(traces)
    if not joined:
        return ""
    if len(joined) <= max_chars:
        return joined
    head_keep = int(max_chars * 0.7)
    tail_keep = int(max_chars * 0.2)
    head = joined[:head_keep]
    tail = joined[-tail_keep:]
    return f"{head}\n... [TRUNCATED {len(joined) - head_keep - tail_keep} CHARS] ...\n{tail}"

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def ensure_local_dir(path: str, what: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return p

def llm_safe_init(**kwargs):
    from vllm import LLM  # local import
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

def build_messages(question: str, answer: str, traces_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": PRESET_SYSTEM},
        {"role": "user", "content": PRESET_USER_WITH_TRACES.format(
            question=question, answer=answer, traces=traces_text
        )},
    ]

def build_prompt_str(tokenizer, question: str, answer: str, traces_text: str) -> str:
    messages = build_messages(question, answer, traces_text)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    # 数据集与模型均为本地路径
    ap.add_argument("--dataset_dir", default="/root/autodl-fs/datasets/reasoning-v1-20m",
                    help="本地数据集目录（glaiveai/reasoning-v1-20m 已下载目录）")
    ap.add_argument("--split", default="default",
                    help="子集名称（该数据集通常为 default）")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
                    help="本地模型目录（如 /root/autodl-tmp/models/Qwen2.5-3B-Instruct）")

    # 生成/运行参数
    ap.add_argument("--engine", default="vllm", choices=["vllm", "dummy"])
    ap.add_argument("--use_vllm_chat", action="store_true",
                    help="直接传 messages 给 vLLM，自动套 chat template")
    ap.add_argument("--dtype", default="bfloat16",
                    help="auto|float16|bfloat16|float32（vLLM 尽量匹配）")
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--limit", type=int, default=2000,
                    help="仅处理前 N 条（大数据集建议先小样本）")
    ap.add_argument("--out", default="/root/autodl-fs/out2000/A+Q+RT/general/out_glaive_answer_summaries.jsonl")
    ap.add_argument("--offline", action="store_true",
                    help="设置 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE=1")

    # 新增：轨迹拼接与预算控制
    ap.add_argument("--traces_max_chars", type=int, default=6000,
                    help="拼入提示的推理轨迹最大字符数（过长将智能截断）")
    ap.add_argument("--prompt_overhead_tokens", type=int, default=128,
                    help="系统/模板等提示的保守 token 预留，用于长度过滤")

    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    ds_dir = str(ensure_local_dir(args.dataset_dir, "Dataset dir"))
    model_dir = str(ensure_local_dir(args.model, "Model dir"))

    print(f"[info] Loading local dataset: {ds_dir} (split={args.split})", file=sys.stderr)
    ds = load_dataset(ds_dir, args.split, split="train", streaming=True)  # 读取本地，不访问网络

    # 采样
    iter_rows = (row for row in ds)
    rows = []
    for i, row in enumerate(iter_rows):
        if args.limit and i >= args.limit:
            break
        rows.append(row)

    # tokenizer（用于长度估计和统计）
    tok_for_len = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def _tok_len(text: str) -> int:
        # 兼容 fast/slow tokenizer
        return len(tok_for_len(text or "", add_special_tokens=False)["input_ids"])

    # 过滤预算 & 准备输入
    overhead_tokens = args.prompt_overhead_tokens
    skipped = []

    problems, final_answers, traces_list, traces_used_list, raw_responses, sample_ids = [], [], [], [], [], []
    t_used_counts, t_raw_counts, t_segments = [], [], []

    for i, row in enumerate(rows):
        q = (row.get("prompt") or "").strip()
        resp = (row.get("response") or "").strip()
        visible_ans = extract_final_answer(resp)
        traces = extract_traces_from_response(resp)

        # 轨迹文本（原始拼接 & 用于提示的截断版）
        traces_raw_joined = join_traces(traces)
        traces_used = format_traces_for_prompt(traces, max_chars=args.traces_max_chars)

        # token 预算估计
        q_len = _tok_len(q)
        a_len = _tok_len(visible_ans)
        t_used_len = _tok_len(traces_used)
        t_raw_len = _tok_len(traces_raw_joined)

        total_len = overhead_tokens + q_len + a_len + t_used_len + args.max_tokens
        if total_len > args.max_model_len:
            skipped.append({
                "sample_id": i,
                "q_len": q_len,
                "a_len": a_len,
                "trace_used_len": t_used_len,
                "total_with_overhead_plus_gen": total_len,
                "max_model_len": args.max_model_len,
                "note": "filtered by length budget (Q+A+traces_used)"
            })
            continue

        # 收集
        problems.append(q)
        final_answers.append(visible_ans)
        traces_list.append(traces)
        traces_used_list.append(traces_used)
        raw_responses.append(resp)
        sample_ids.append(i)
        t_used_counts.append(t_used_len)
        t_raw_counts.append(t_raw_len)
        t_segments.append(len(traces))

    # 写出被跳过样本
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_log = out_path.with_suffix(".skipped.jsonl")
    skipped_log.parent.mkdir(parents=True, exist_ok=True)  # 保险：若与主目录不同也创建
    with open(skipped_log, "w", encoding="utf-8") as f:
        for rec in skipped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[filter] kept={len(problems)} skipped={len(skipped)} (log: {skipped_log})", file=sys.stderr)

    # ==== 构造输入 ====
    inputs_messages = [build_messages(q, a, t) for q, a, t in zip(problems, final_answers, traces_used_list)]
    tokenizer = None

    # ==== 生成 ====
    if args.engine == "vllm":
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not installed. `pip install vllm` or use --engine dummy.")

        llm_kwargs = dict(model=model_dir, trust_remote_code=True)
        for k, v in dict(dtype=args.dtype,
                         gpu_memory_utilization=args.gpu_mem_util,
                         max_model_len=args.max_model_len).items():
            if v is not None:
                llm_kwargs[k] = v

        print(f"[info] Initializing vLLM with {llm_kwargs}", file=sys.stderr)
        llm = llm_safe_init(**llm_kwargs)
        sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        summaries: List[str] = []
        if args.use_vllm_chat:
            try:
                for batch in chunked(inputs_messages, args.batch_size):
                    outs = llm.generate(batch, sampling)
                    for o in outs:
                        text = o.outputs[0].text if o.outputs else ""
                        summaries.append(text.strip())
            except Exception as e:
                print(f"[warn] messages path failed ({e}); fallback to apply_chat_template.", file=sys.stderr)
                args.use_vllm_chat = False

        if not args.use_vllm_chat:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            inputs_str = [
                build_prompt_str(tokenizer, q, a, t)
                for q, a, t in zip(problems, final_answers, traces_used_list)
            ]
            for batch in chunked(inputs_str, args.batch_size):
                outs = llm.generate(batch, sampling)
                for o in outs:
                    text = o.outputs[0].text if o.outputs else ""
                    summaries.append(text.strip())
    else:
        summaries = [""] * len(problems)

    # ==== 写出 ====
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sid, q, a, s, traces, t_used_text, t_used, t_raw, segs, resp_raw in zip(
            sample_ids, problems, final_answers, summaries,
            traces_list, traces_used_list, t_used_counts, t_raw_counts, t_segments, raw_responses
        ):
            rec = {
                "sample_id": sid,
                "question": q,
                "final_answer": a,                 # 仅可见答案（去除 <think>）
                # 名称保持兼容：虽名为 answer_driven，但现在已注入 traces 做增强
                "summary_answer_driven": s,
                "traces_natural": traces,          # 原始自然推理轨迹（list）
                "traces_used_in_prompt": t_used_text,  # 实际拼入提示（可能被截断）
                "response_raw": resp_raw,          # 原始整段 response
                "meta": {
                    "source_dataset": "glaiveai/reasoning-v1-20m",
                    "engine": args.engine,
                    "model": model_dir,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_tokens,
                    "dtype": args.dtype,
                    "gpu_memory_utilization": args.gpu_mem_util,
                    "max_model_len": args.max_model_len,
                    "use_vllm_chat": bool(args.use_vllm_chat),
                    "offline": bool(args.offline),
                    # 轨迹统计
                    "trace_segments": segs,
                    "trace_tokens_used": t_used,    # 用于提示（截断后）的 token 数
                    "trace_tokens_raw": t_raw,      # 原始完整轨迹 token 数
                    "traces_max_chars": args.traces_max_chars,
                    "prompt_overhead_tokens": args.prompt_overhead_tokens,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[ok] Wrote {len(summaries)} records to {str(out_path)}", file=sys.stderr)


if __name__ == "__main__":
    main()
