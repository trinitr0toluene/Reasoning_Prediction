# -*- coding: utf-8 -*-
"""
data_processing.py
------------------
(1) 合并多源 JSONL/Parquet → 统一 JSONL（原功能，保留）
(2) 直接从 HuggingFace datasets 拉取 → 解析 <think> → 计数 → 生成统一或“prompt/response”格式（新增）

依赖：
- JSONL/基础功能：标准库
- 读 Parquet：pip install pyarrow
- 从 HF 拉数/计 token：pip install datasets transformers
"""

import os
import glob
import json
import argparse
import hashlib
import re
from typing import Dict, Any, Iterable, List, Tuple, Optional

# ---------- 可选依赖 ----------
try:
    import pyarrow.dataset as ds  # for parquet
except Exception:
    ds = None

# 懒加载：datasets / transformers 在用到时再 import


# ============== 通用 I/O ==============

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl_stream(path: str):
    f = open(path, "w", encoding="utf-8")
    def _write(obj: Dict[str, Any]):
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return f, _write


# ============== Parquet 读取 ==============

def iter_parquet_rows(pattern_or_file: str) -> Iterable[Dict[str, Any]]:
    if ds is None:
        raise ImportError("pyarrow 未安装；读取 .parquet 需先 `pip install pyarrow`。")
    files = sorted(glob.glob(pattern_or_file)) if any(ch in pattern_or_file for ch in "*?[]") else [pattern_or_file]
    if not files:
        return
    dataset = ds.dataset(files, format="parquet")
    for batch in dataset.to_batches(batch_size=4096):
        cols = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
        n = batch.num_rows
        for i in range(n):
            yield {k: cols[k][i].as_py() for k in cols}


def iter_sources(paths: List[str]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        has_glob = any(ch in p for ch in "*?[]")
        files = sorted(glob.glob(p)) if has_glob else [p]
        if not files:
            continue
        ext = os.path.splitext(files[0])[1].lower()
        if ext == ".parquet":
            yield from iter_parquet_rows(p if has_glob else files[0])
        else:
            for f in files:
                yield from read_jsonl(f)


# ============== Join key / Summary ==============

def make_key(d: Dict[str, Any]) -> Tuple:
    if "uuid" in d and "gen_index" in d: return ("uuid+gen", str(d["uuid"]), int(d["gen_index"]))
    if "sample_id" in d and "gen_index" in d: return ("sid+gen", str(d["sample_id"]), int(d["gen_index"]))
    if "id" in d: return ("id", str(d["id"]))
    if "qid" in d: return ("qid", str(d["qid"]))
    if "uuid" in d: return ("uuid", str(d["uuid"]))
    if "sample_id" in d: return ("sid", str(d["sample_id"]))
    q = d.get("prompt", d.get("question", ""))
    return ("hash", hashlib.md5(q.encode("utf-8")).hexdigest())

def extract_summary_text(d: Dict[str, Any]) -> str:
    for k in ("summary", "summary_text", "summary_answer_driven"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def build_summary_index(path: str) -> Dict[Tuple, str]:
    if not path: return {}
    idx: Dict[Tuple, str] = {}
    for row in read_jsonl(path):
        s = extract_summary_text(row)
        if s:
            idx[make_key(row)] = s
    return idx


# ============== 文本工具 ==============

def join_text(x: Any) -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, list): return "\n".join(join_text(t) for t in x)
    if isinstance(x, dict):
        if "content" in x: return join_text(x["content"])
        return "\n".join(f"{k}: {join_text(v)}" for k, v in x.items())
    return str(x)

_PROMPT_KEYS = [
    "prompt", "question", "query", "problem", "instruction",
    "input", "user_input", "task", "problem_text", "question_text"
]
_ANS_KEYS = [
    "final_answer", "answer", "output", "response", "final",
    "gt_answer", "target", "solution_final"
]

def _pick_first_str(d, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def extract_prompt_generic(r: dict) -> str:
    v = _pick_first_str(r, _PROMPT_KEYS)
    if v: return v
    msgs = r.get("messages") or r.get("conversations") or r.get("dialog") or r.get("chat")
    if isinstance(msgs, list) and msgs:
        user_parts = []
        for m in msgs:
            role = (m.get("role") or m.get("speaker") or "").lower()
            content = m.get("content") or m.get("text") or m.get("value")
            if role in ("user", "human"):
                user_parts.append(join_text(content))
        if user_parts:
            return "\n".join([p for p in user_parts if p.strip()]).strip()
        first = msgs[0].get("content") if isinstance(msgs[0], dict) else msgs[0]
        if first:
            return join_text(first).strip()
    meta = r.get("meta") or r.get("metadata")
    if isinstance(meta, dict):
        v = _pick_first_str(meta, _PROMPT_KEYS)
        if v: return v
    return ""

def extract_answer_generic(r: dict) -> str:
    v = _pick_first_str(r, _ANS_KEYS)
    if v: return v
    msgs = r.get("messages") or r.get("conversations") or r.get("dialog") or r.get("chat")
    if isinstance(msgs, list) and msgs:
        cand = None
        for m in msgs:
            role = (m.get("role") or m.get("speaker") or "").lower()
            content = m.get("content") or m.get("text") or m.get("value")
            if role in ("assistant", "bot"):
                cand = content
        if cand is not None:
            return join_text(cand).strip()
        last = msgs[-1].get("content") if isinstance(msgs[-1], dict) else msgs[-1]
        if last:
            return join_text(last).strip()
    meta = r.get("meta") or r.get("metadata")
    if isinstance(meta, dict):
        v = _pick_first_str(meta, _ANS_KEYS)
        if v: return v
    return ""


# ============== <think> 解析 ==============

_THINK_OPEN = re.compile(r"<think>", re.S)
_THINK_CLOSE = re.compile(r"</think>", re.S)

def extract_think_and_rest(text: str) -> Tuple[List[str], str]:
    """抓取所有 <think>...</think>，并返回紧随最后一个 </think> 之后的剩余文本。"""
    if not isinstance(text, str) or not text:
        return [], ""
    thinks = re.findall(r"<think>(.*?)</think>", text, flags=re.S)
    last_end = 0
    for m in _THINK_CLOSE.finditer(text):
        last_end = m.end()
    rest = text[last_end:].strip() if last_end else text.strip()
    return thinks, rest


# ============== 统一 schema 归一化（原通道） ==============

def normalize_record(
    r: Dict[str, Any],
    summary_idx: Optional[Dict[Tuple, str]],
    attach_summary: bool,
    infer_missing_y: bool = False,
    infer_fields: Optional[List[str]] = None,
    tokenizer=None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "uuid": r.get("uuid"),
        "gen_index": r.get("gen_index"),
        "sample_id": r.get("sample_id"),
        "id": r.get("id"),
        "qid": r.get("qid"),
        "prompt": extract_prompt_generic(r),
        "final_answer": extract_answer_generic(r),
        "reasoning_token_count": r.get(
            "reasoning_token_count",
            r.get("reasoning_tokens", r.get("reasoning_len", r.get("hidden_tokens", r.get("hidden_len")))),
        ),
    }
    if attach_summary and summary_idx:
        s = summary_idx.get(make_key(r))
        if s: out["summary_text"] = s
    if "domain" in r: out["domain"] = r["domain"]
    if "task_type" in r and "domain" not in out: out["domain"] = r["task_type"]
    if "provider_id" in r: out["provider_id"] = r["provider_id"]

    if out["reasoning_token_count"] is None and infer_missing_y and tokenizer is not None:
        infer_fields = infer_fields or ["think","trace","rationale","reasoning","chain_of_thought","solution","thoughts"]
        for k in infer_fields:
            if k in r and r[k] is not None:
                text = join_text(r[k])
                try:
                    out["reasoning_token_count"] = int(len(tokenizer.encode(text)))
                    break
                except Exception:
                    pass
    return out


# ============== HF 通道：直接拉数据集并生成 ==============

def hf_iter_rows(name: str, split: str, hf_subset: Optional[str] = None, limit: Optional[int] = None):
    from datasets import load_dataset
    ds = load_dataset(name, hf_subset, split=split) if hf_subset else load_dataset(name, split=split)
    if limit is not None:
        for i in range(min(limit, len(ds))):
            yield ds[i]
    else:
        for row in ds:
            yield row

def build_from_hf_row(
    row: Dict[str, Any],
    tk,                                  # transformers tokenizer
    question_key: str,
    answer_key: str,
    reasoning_key: Optional[str],
    require_think: bool,
    join_mode: str,                       # "unified" or "count-supervision"
    include_io_counts: bool = True,
) -> Optional[Dict[str, Any]]:
    # 取题面/答案
    problem = (row.get(question_key) or row.get("question") or row.get("prompt") or "").strip()
    answer_raw = (row.get(answer_key) or row.get("response") or row.get("answer") or row.get("output") or "")
    answer_raw = join_text(answer_raw).strip()

    # 从 answer_raw 里解析 <think>
    thinks, rest = extract_think_and_rest(answer_raw)

    # 如果单独提供了 reasoning 字段，则追加
    if (not thinks) and reasoning_key and reasoning_key in row and row[reasoning_key] is not None:
        rk = row[reasoning_key]
        rtxt = join_text(rk).strip()
        if rtxt:
            thinks = [rtxt]

    if require_think and not thinks:
        return None

    # 计算 token 数
    in_tok = len(tk.tokenize(problem)) if include_io_counts else 0
    out_tok = len(tk.tokenize(rest)) if include_io_counts else 0
    reason_tok = len(tk.tokenize("\n".join(thinks))) if thinks else 0

    if join_mode == "count-supervision":
        # 生成与你脚本一致的 prompt/response
        prompt = f"Problem: {problem}\nAnswer: {rest}\n"
        if include_io_counts:
            prompt += f"The problem has {in_tok} tokens, and the answer has {out_tok} tokens."
        resp = f"{reason_tok}."
        return {"prompt": prompt, "response": resp}

    # 否则生成统一 schema，供回归/GRPO 管线
    rec = {
        "uuid": row.get("uuid"),
        "gen_index": row.get("gen_index"),
        "sample_id": row.get("sample_id"),
        "id": row.get("id"),
        "qid": row.get("qid"),
        "prompt": problem,
        "final_answer": rest,
        "reasoning_token_count": reason_tok if thinks else None,
    }
    if "domain" in row: rec["domain"] = row["domain"]
    if "task_type" in row and "domain" not in rec: rec["domain"] = row["task_type"]
    return rec


# ============== CLI ==============

def parse_args():
    ap = argparse.ArgumentParser()

    # 互斥的两种输入方式：本地多源 / HuggingFace 数据集
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sources", nargs="+", help="JSONL/Parquet 路径或通配符")
    src.add_argument("--hf-dataset", help="HuggingFace 数据集名（如 'FreedomIntelligence/Medical-R1-Distill-Data'）")

    # HF 专属参数
    ap.add_argument("--hf-subset", default=None, help="HuggingFace 子集名（可选）")
    ap.add_argument("--hf-split", default="train", help="train/validation/test 等")
    ap.add_argument("--hf-question-key", default="question", help="题面字段名，默认 'question'")
    ap.add_argument("--hf-answer-key", default="response", help="带 <think> 的回答字段名，默认 'response'")
    ap.add_argument("--hf-reasoning-key", default=None, help="若推理在独立字段（如 'reasoning (reasoning_content)'），填这里")
    ap.add_argument("--hf-require-think", action="store_true", help="若开启，则无 <think> 的样本会被跳过")
    ap.add_argument("--hf-limit", type=int, default=None, help="仅拉取前 N 条（调试用）")

    # 输出控制
    ap.add_argument(
        "--output",
        required=False,
        default="/root/autodl-fs/openr1_math_all.jsonl",
        help="输出 JSONL 路径",
    )
    ap.add_argument("--emit", choices=["unified", "count-supervision"], default="unified",
                    help="unified: prompt/final_answer/reasoning_token_count；count-supervision: prompt/response")
    ap.add_argument("--attach-summary", action="store_true", help="合并 summary_text（仅对 --sources 流程有意义）")
    ap.add_argument("--summary-file", default="", help="summary JSONL（仅对 --sources 流程有意义）")

    # 目标值推断（仅 --sources 流程）
    ap.add_argument("--infer-missing-y", action="store_true",
                    help="当样本缺 reasoning_token_count 时，尝试从 trace 字段估算 token 数")
    ap.add_argument("--infer-fields", default="think,trace,rationale,reasoning,chain_of_thought,solution,thoughts")
    ap.add_argument("--skip-missing-y", action="store_true", help="推断后仍缺 y 的样本是否丢弃")

    # tokenizer（两条流程都会用到）
    ap.add_argument("--tokenizer", default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
                    help="用于 token 计数的 HF tokenizer 名；示例：deepseek-ai/DeepSeek-V3-0324")

    return ap.parse_args()


def main():
    args = parse_args()

    # 懒加载 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    # 输出 writer
    fout, write_line = write_jsonl_stream(args.output)
    total = kept = miss_y = 0

    try:
        if args.hf_dataset:
            # 直接从 HuggingFace 拉取并构建
            for row in hf_iter_rows(args.hf_dataset, args.hf_split, args.hf_subset, args.hf_limit):
                rec = build_from_hf_row(
                    row=row,
                    tk=tokenizer,
                    question_key=args.hf_question_key,
                    answer_key=args.hf_answer_key,
                    reasoning_key=args.hf_reasoning_key,
                    require_think=args.hf_require_think,
                    join_mode=args.emit,
                    include_io_counts=True,
                )
                if rec is None:
                    continue
                # 如果 emit=unified 且 y 缺失，按选项决定保留与否
                if args.emit == "unified" and rec.get("reasoning_token_count") is None:
                    miss_y += 1
                    if args.skip_missing_y:
                        continue
                write_line(rec); kept += 1; total += 1

        else:
            # 本地多源 → 统一 JSONL
            sidx = build_summary_index(args.summary_file)
            infer_fields: List[str] = [s.strip() for s in args.infer_fields.split(",") if s.strip()]

            for raw in iter_sources(args.sources):
                rec = normalize_record(
                    raw, sidx, args.attach_summary,
                    infer_missing_y=args.infer_missing_y,
                    infer_fields=infer_fields,
                    tokenizer=tokenizer
                )
                total += 1
                if rec.get("reasoning_token_count") is None:
                    miss_y += 1
                    if args.skip_missing_y:
                        continue
                # 若用户选择 emit=count-supervision，则把 unified 转为 prompt/response 形式
                if args.emit == "count-supervision":
                    prob = rec.get("prompt", "")
                    ans = rec.get("final_answer", "")
                    in_tok = len(tokenizer.tokenize(prob))
                    out_tok = len(tokenizer.tokenize(ans))
                    y = rec.get("reasoning_token_count") or 0
                    prompt = f"Problem: {prob}\nAnswer: {ans}\nThe problem has {in_tok} tokens, and the answer has {out_tok} tokens."
                    rec = {"prompt": prompt, "response": f"{y}."}
                write_line(rec); kept += 1

    finally:
        fout.close()

    print(f"[info] input_rows≈{total}  written_rows={kept}  missing_y_after={miss_y}  -> {args.output}")


if __name__ == "__main__":
    main()
