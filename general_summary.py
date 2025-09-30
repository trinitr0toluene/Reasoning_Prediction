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

# 通用摘要预设（适用于通用推理任务）
PRESET_SYSTEM = (
    "You are a concise assistant. Given a Question and its Final Answer, "
    "produce a short, faithful explanation as 3–5 bullet points. "
    "Avoid full step-by-step derivations; focus on key ideas, pivotal transformations, "
    "and a brief verification."
)
PRESET_USER = "Question:\n{question}\n\nFinal Answer:\n{answer}\n\nGive 3–5 bullet points."

# 提取 <think>…</think>
THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
# 常见 “Final Answer:” 标记（英文/中文）
FINAL_ANS_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:final\s*answer|答案)\s*[:：]?\s*(.+)$",
               re.IGNORECASE | re.DOTALL),
]

# ===================== Utilities =====================

def strip_think(resp: str) -> str:
    """移除 <think>…</think> 段，得到用户可见部分。"""
    return THINK_TAG_RE.sub("", resp or "").strip()

def extract_final_answer(resp: str) -> str:
    """从 response 中提取可见 final answer：
       1) 去掉 <think>；
       2) 若存在 'Final Answer:' 等标记，取其后文本；
       3) 否则取最后一行非空文本。"""
    vis = strip_think(resp)
    for pat in FINAL_ANS_PATTERNS:
        m = pat.search(vis)
        if m:
            return m.group(1).strip()
    # 退路：取末行
    lines = [ln.strip() for ln in vis.splitlines() if ln.strip()]
    return lines[-1] if lines else vis

def extract_traces_from_response(resp: str) -> List[str]:
    if not isinstance(resp, str) or not resp:
        return []
    return [m.strip() for m in THINK_TAG_RE.findall(resp)]

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def ensure_local_dir(path: str, what: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return p

def llm_safe_init(**kwargs):
    """vLLM 版本兼容：剔除不支持的 kw。"""
    from vllm import LLM  # import 本地作用域
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

def build_messages(question: str, answer: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": PRESET_SYSTEM},
        {"role": "user", "content": PRESET_USER.format(question=question, answer=answer)},
    ]

def build_prompt_str(tokenizer, question: str, answer: str) -> str:
    messages = build_messages(question, answer)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    # 数据集与模型均为本地路径
    ap.add_argument("--dataset_dir", default = "/root/autodl-fs/datasets/reasoning-v1-20m",
                    help="本地数据集目录（glaiveai/reasoning-v1-20m 已下载目录）")
    ap.add_argument("--split", default="default",
                    help="子集名称（该数据集通常为 default）")
    ap.add_argument("--model", default = "/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
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

    ap.add_argument("--limit", type=int, default=20000,
                    help="仅处理前 N 条（大数据集建议先小样本）")
    ap.add_argument("--out", default="/root/autodl-fs/out_glaive_answer_summaries.jsonl")
    ap.add_argument("--offline", action="store_true",
                    help="设置 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE=1")

    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    ds_dir = str(ensure_local_dir(args.dataset_dir, "Dataset dir"))
    model_dir = str(ensure_local_dir(args.model, "Model dir"))

    print(f"[info] Loading local dataset: {ds_dir} (split={args.split})", file=sys.stderr)
    ds = load_dataset(ds_dir, args.split, split="train", streaming=True)  # 读取本地，不访问网络

    # if args.limit and args.limit > 0:
    #     ds = ds.select(range(min(args.limit, len(ds))))
    iter_rows = (row for row in ds)
    rows = []
    for i, row in enumerate(iter_rows):
        if args.limit and i >= args.limit:
            break
        rows.append(row)


    # 字段：glaive 数据集为 prompt / response
    problems, final_answers, traces_list, raw_responses, sample_ids = [], [], [], [], []
    for i, row in enumerate(rows):
        q = (row.get("prompt") or "").strip()
        resp = (row.get("response") or "").strip()
        traces = extract_traces_from_response(resp)
        visible_ans = extract_final_answer(resp)
        problems.append(q)
        final_answers.append(visible_ans)
        traces_list.append(traces)
        raw_responses.append(resp)
        sample_ids.append(i)

    # ==== 构造输入 ====
    inputs_messages = [build_messages(q, a) for q, a in zip(problems, final_answers)]
    tokenizer = None

    # ==== 生成 ====
    if args.engine == "vllm":
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not installed. `pip install vllm` or use --engine dummy.")

        llm_kwargs = dict(model=model_dir, trust_remote_code=True)
        # 这些 kw 视 vLLM 版本而定，不支持会自动剔除
        for k, v in dict(dtype=args.dtype,
                         gpu_memory_utilization=args.gpu_mem_util,
                         max_model_len=args.max_model_len).items():
            if v is not None:
                llm_kwargs[k] = v

        print(f"[info] Initializing vLLM with {llm_kwargs}", file=sys.stderr)
        llm = llm_safe_init(**llm_kwargs)
        sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        summaries: List[str] = []
        # 优先 messages 直传
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

        # 回退到字符串 prompts
        if not args.use_vllm_chat:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            inputs_str = [build_prompt_str(tokenizer, q, a) for q, a in zip(problems, final_answers)]
            for batch in chunked(inputs_str, args.batch_size):
                outs = llm.generate(batch, sampling)
                for o in outs:
                    text = o.outputs[0].text if o.outputs else ""
                    summaries.append(text.strip())
    else:
        # 占位：不做真实生成，便于先打通管线
        summaries = [""] * len(problems)

    # ==== 写出 ====
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for sid, q, a, s, traces, resp_raw in zip(
            sample_ids, problems, final_answers, summaries, traces_list, raw_responses
        ):
            rec = {
                "sample_id": sid,
                "question": q,
                "final_answer": a,              # 仅可见答案（去除 <think>）
                "summary_answer_driven": s,     # 仅基于 Q + final_answer 生成
                "traces_natural": traces,       # 自然推理轨迹（<think>…</think>）
                "response_raw": resp_raw,       # 原始整段 response
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
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[ok] Wrote {len(summaries)} records to {str(out_path)}", file=sys.stderr)


if __name__ == "__main__":
    main()
