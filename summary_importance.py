# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm

# --------------------
# Utility: JSONL IO
# --------------------
def read_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def write_jsonl(path: str, records: Iterable[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# --------------------
# Data structures
# --------------------
@dataclass(frozen=True)
class Key:
    uuid: str
    gen_index: int

@dataclass
class Sample:
    key: Key
    problem: str
    gold_answer: str
    steps: List[str]         # steps according to chosen granularity
    full_trace: str          # concatenation of steps

@dataclass
class GenTask:
    sample_key: Key
    variant: str       # "base" or f"drop_{k}"
    step_index: int    # -1 for base; otherwise dropped step index
    prompt_text: str

# --------------------
# Answer extraction & normalization
# --------------------
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
ANS_TAG_RE = re.compile(r"(?:Final Answer|Answer)\s*[:：]\s*(.+)", re.IGNORECASE)

def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = ANS_TAG_RE.search(text)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text.strip()

LATEX_CLEAN_RE = re.compile(
    r"(\\mathrm\{[^}]*\}|\\text\{[^}]*\}|\\left|\\right|\\!|\\,|\\;|\\:|\\quad|\\qquad|\\hspace\{[^}]*\}|\\phantom\{[^}]*\})"
)
MATH_MARK_RE = re.compile(r"[\$\u200b]")
WS_RE = re.compile(r"\s+")

def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = LATEX_CLEAN_RE.sub("", s)
    s = MATH_MARK_RE.sub("", s)
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    s = WS_RE.sub("", s)
    return s

# --------------------
# Prompt builder (Qwen chat)
# --------------------
SYSTEM_PROMPT = (
    "You are a precise math solver. Use the provided hints to compute the answer. "
    "Do NOT show steps. Respond with only the final answer, ideally in LaTeX like \\boxed{...}."
)

USER_TEMPLATE = """Solve the problem using the hints below. Do not explain your steps.

[Problem]
{problem}

[Hints]
{hints}

Respond ONLY with the final answer (e.g., \\boxed{{...}}).
"""

def build_chat_prompt(tokenizer, problem: str, hints: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(problem=problem, hints=hints)},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# --------------------
# Sentence / step segmentation from summary
# --------------------
import regex as re2
RE_LATEX_BLOCK = re2.compile(r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\))", flags=re2.S)
RE_BULLET = re2.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s+")
RE_SENT_END = re2.compile(r"([\.!?])(\s+|$)")
RE_META = re2.compile(r"^\s*(ok(ay)?|hmm+|let me|now|wait|note that|remember|i need to|i should|maybe|well|alright)\b", re2.I)

def _normalize_text(s: str) -> str:
    s = s.replace("\r","\n")
    s = re2.sub(r"[ \t]+"," ", s)
    s = re2.sub(r"\n{3,}","\n\n", s)
    return s.strip()

def _hold_latex(s: str) -> Tuple[str,List[str]]:
    blocks=[]
    def repl(m):
        blocks.append(m.group(0))
        return f"⟦LATEX_{len(blocks)-1}⟧"
    return RE_LATEX_BLOCK.sub(repl, s), blocks

def _restore_latex(s: str, blocks: List[str]) -> str:
    for i,b in enumerate(blocks):
        s = s.replace(f"⟦LATEX_{i}⟧", b)
    return s

def split_sentences_from_text(text: str) -> List[str]:
    """保护 LaTeX，按行号/句末符分句，并把极短口癖句并到后句。"""
    text = _normalize_text(text or "")
    text_hold, blocks = _hold_latex(text)

    rough=[]
    for line in text_hold.split("\n"):
        line=line.strip()
        if not line: continue
        if RE_BULLET.match(line):
            rough.append(RE_BULLET.sub("", line)); continue
        start=0
        for m in RE_SENT_END.finditer(line):
            end=m.end(1)
            rough.append(line[start:end].strip())
            start=m.end()
        tail=line[start:].strip()
        if tail: rough.append(tail)

    rough=[_restore_latex(s, blocks) for s in rough]

    merged=[]; i=0
    while i<len(rough):
        seg=rough[i].strip()
        if not seg: i+=1; continue
        if (len(seg)<=6 or RE_META.match(seg)) and i+1 < len(rough):
            merged.append((seg + " " + rough[i+1]).strip()); i += 2
        else:
            merged.append(seg); i += 1
    return merged

# --------------------
# Data loading & alignment
# --------------------
def align_from_preprocessed(
    raw_masked_path: str,
    macro_steps_path: str,
    sentences_path: str,
    granularity: str = "macro",
) -> List[Sample]:
    raw = read_jsonl(raw_masked_path)
    raw_by_key: Dict[Key, dict] = {}
    for r in raw:
        raw_by_key[Key(uuid=r["uuid"], gen_index=int(r["gen_index"]))] = r

    if granularity == "macro":
        seg = read_jsonl(macro_steps_path)
        get_steps = lambda obj: [ms["text"] for ms in obj["macro_steps"]]
    elif granularity == "sentence":
        seg = read_jsonl(sentences_path)
        get_steps = lambda obj: [s["text"] for s in obj["sentences"]]
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    samples: List[Sample] = []
    for s in seg:
        key = Key(uuid=s["uuid"], gen_index=int(s["gen_index"]))
        if key not in raw_by_key:
            continue
        r = raw_by_key[key]
        steps = get_steps(s)
        full_trace = "\n".join(steps)
        samples.append(
            Sample(
                key=key,
                problem=r.get("problem",""),
                gold_answer=r.get("answer_gold","") or r.get("answer",""),
                steps=steps,
                full_trace=full_trace,
            )
        )
    return samples

def align_from_summary_jsonl(
    summary_jsonl: str,
    granularity: str = "sentence",
    source_priority: Tuple[str,...] = ("summary_answer_driven","traces_natural"),
) -> List[Sample]:
    """直接从 /mnt/data/out_openr1_answer_summaries.jsonl 构造样本。"""
    data = read_jsonl(summary_jsonl)
    samples: List[Sample] = []
    for i, ex in enumerate(data):
        uuid = ex.get("uuid") or ex.get("id") or f"rec_{i:06d}"
        problem = ex.get("question","")
        gold_answer = ex.get("final_answer","") or ex.get("answer","") or ""
        text_src = None

        for key in source_priority:
            val = ex.get(key)
            if isinstance(val, str) and val.strip():
                text_src = val.strip()
                break
            if isinstance(val, list) and val:
                text_src = "\n".join([str(v) for v in val if str(v).strip()])
                break

        if not text_src:
            # 没有 summary 也没有 traces，就跳过
            continue

        if granularity == "sentence":
            steps = split_sentences_from_text(text_src)
        elif granularity == "macro":
            # 简约做法：仍按句子切，但后续你可以替换为更强的 chunker
            steps = split_sentences_from_text(text_src)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

        full_trace = "\n".join(steps)
        samples.append(Sample(
            key=Key(uuid=uuid, gen_index=0),  # summary 默认只有一条轨迹
            problem=problem,
            gold_answer=gold_answer,
            steps=steps,
            full_trace=full_trace,
        ))
    return samples

# --------------------
# Batching helper
# --------------------
def batched(iterable: Iterable, n: int) -> Iterable[List]:
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

# --------------------
# Core: run ablation
# --------------------
def run_ablation(
    model_name: str,
    granularity: str,
    out_path: str,
    # —— data sources ——
    summary_jsonl: Optional[str] = None,
    raw_masked_path: Optional[str] = None,
    macro_steps_path: Optional[str] = None,
    sentences_path: Optional[str] = None,
    # —— gen params ——
    max_new_tokens: int = 32,
    bsz: int = 32,
    limit: Optional[int] = None,
    dtype: str = "float16",
    gpu_mem_util: float = 0.75,
    max_model_len: int = 16384,
    download_dir: Optional[str] = None,
    offline: bool = True,
    ctx_headroom: int = 64,
    importance: str = "logp",              # or "acc_drop"
    normalize: str = "minmax",
    stop: Optional[List[str]] = None,
    tp: int = 1,
):
    # --- env for offline & threads ---
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if download_dir is None:
        download_dir = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

    local_only = offline or os.path.isdir(model_name)

    print(f"Loading tokenizer for {model_name} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_only,
    )

    print(f"Loading model {model_name} (dtype={dtype}) with vLLM ...", file=sys.stderr)
    llm = LLM(
        model=model_name,
        dtype=dtype,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        download_dir=download_dir,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=stop or [],
        logprobs=1,
    )

    # ---------- build samples ----------
    if summary_jsonl:
        print(f"[data] Using summary JSONL: {summary_jsonl}", file=sys.stderr)
        samples = align_from_summary_jsonl(summary_jsonl, granularity=granularity)
    else:
        print(f"[data] Using preprocessed files", file=sys.stderr)
        if not (raw_masked_path and macro_steps_path and sentences_path):
            raise ValueError("When summary_jsonl is not provided, raw/macro/sentences paths are required.")
        samples = align_from_preprocessed(
            raw_masked_path=raw_masked_path,
            macro_steps_path=macro_steps_path,
            sentences_path=sentences_path,
            granularity=granularity,
        )

    if limit is not None:
        samples = samples[:limit]
    print(f"Total aligned samples: {len(samples)}", file=sys.stderr)

    # ---------- helper: token counting & truncation ----------
    vtok = llm.get_tokenizer()

    def prompt_len(text: str) -> int:
        return len(vtok.encode(text))

    def fit_head_tail(problem: str, steps: List[str], keep_idx: int, budget: int, sep: str = "\n"):
        if keep_idx == -1:
            other = steps
        else:
            other = steps[:keep_idx] + steps[keep_idx+1:]

        empty_prompt = build_chat_prompt(tokenizer, problem, "")
        if prompt_len(empty_prompt) > budget:
            return "", 0, 0, prompt_len(empty_prompt)

        head, tail = [], []
        i, j = 0, len(other) - 1

        def build_with(head_ls, tail_ls) -> Tuple[str, int]:
            hints = sep.join(head_ls + (["<...>"] if head_ls or tail_ls else []) + tail_ls)
            pr = build_chat_prompt(tokenizer, problem, hints)
            return hints, prompt_len(pr)

        while i <= j:
            cand_head = head + [other[i]]
            hints, toks = build_with(cand_head, tail)
            if toks <= budget:
                head = cand_head
                i += 1
            else:
                break

        while i <= j:
            cand_tail = [other[j]] + tail
            hints, toks = build_with(head, cand_tail)
            if toks <= budget:
                tail = cand_tail
                j -= 1
            else:
                break

        hints, toks = build_with(head, tail)
        return hints, len(head), len(tail), toks

    CTX_BUDGET = max_model_len - max_new_tokens - ctx_headroom
    if CTX_BUDGET <= 0:
        raise ValueError(
            f"Invalid context budget: max_model_len({max_model_len}) - "
            f"max_new_tokens({max_new_tokens}) - headroom({ctx_headroom}) <= 0"
        )
    print(f"[truncate] context budget={CTX_BUDGET}, headroom={ctx_headroom}", file=sys.stderr)

    trunc_logs: List[dict] = []
    impossible_logs: List[dict] = []

    # ---------- build generation tasks (with truncation) ----------
    tasks: List[GenTask] = []
    gold_by_key = {}

    for s in samples:
        # golds for acc_drop
        gold_by_key[s.key] = s.gold_answer

        base_prompt = build_chat_prompt(tokenizer, s.problem, s.full_trace)
        base_len = prompt_len(base_prompt)
        base_hints = s.full_trace
        truncated_variants = []

        empty_len = prompt_len(build_chat_prompt(tokenizer, s.problem, ""))
        if empty_len > CTX_BUDGET or empty_len > max_model_len:
            impossible_logs.append({
                "uuid": s.key.uuid, "gen_index": s.key.gen_index,
                "reason": "problem_too_long",
                "empty_prompt_tokens": empty_len,
                "ctx_budget": CTX_BUDGET, "max_model_len": max_model_len,
            })
            continue

        if base_len > CTX_BUDGET or base_len > max_model_len:
            ht, hcnt, tcnt, toks_after = fit_head_tail(s.problem, s.steps, keep_idx=-1, budget=CTX_BUDGET)
            base_hints = ht
            base_prompt = build_chat_prompt(tokenizer, s.problem, base_hints)
            truncated_variants.append({
                "variant": "base",
                "before_tokens": base_len, "after_tokens": toks_after,
                "head_steps": hcnt, "tail_steps": tcnt
            })

        tasks.append(GenTask(sample_key=s.key, variant="base", step_index=-1, prompt_text=base_prompt))

        for k in range(len(s.steps)):
            hints_k = "\n".join(s.steps[:k] + s.steps[k+1:])
            prompt_k = build_chat_prompt(tokenizer, s.problem, hints_k)
            len_k = prompt_len(prompt_k)

            if len_k > CTX_BUDGET or len_k > max_model_len:
                ht, hcnt, tcnt, toks_after = fit_head_tail(s.problem, s.steps, keep_idx=k, budget=CTX_BUDGET)
                hints_k = ht
                prompt_k = build_chat_prompt(tokenizer, s.problem, hints_k)
                truncated_variants.append({
                    "variant": f"drop_{k}",
                    "before_tokens": len_k, "after_tokens": toks_after,
                    "head_steps": hcnt, "tail_steps": tcnt
                })

            tasks.append(GenTask(sample_key=s.key, variant=f"drop_{k}", step_index=k, prompt_text=prompt_k))

        if truncated_variants:
            trunc_logs.append({
                "uuid": s.key.uuid,
                "gen_index": s.key.gen_index,
                "granularity": granularity,
                "ctx_budget": CTX_BUDGET,
                "model": str(model_name),
                "max_model_len": max_model_len,
                "max_new_tokens": max_new_tokens,
                "truncated_variants": truncated_variants,
            })

    if trunc_logs:
        trunc_path = os.path.splitext(out_path)[0] + ".truncated.jsonl"
        write_jsonl(trunc_path, trunc_logs)
        print(f"[truncate] Wrote {len(trunc_logs)} truncated-sample records -> {trunc_path}", file=sys.stderr)

    if impossible_logs:
        imp_path = os.path.splitext(out_path)[0] + ".impossible.jsonl"
        write_jsonl(imp_path, impossible_logs)
        print(f"[truncate] Wrote {len(impossible_logs)} impossible-sample records -> {imp_path}", file=sys.stderr)

    print(f"Total generation tasks (after truncation): {len(tasks)}", file=sys.stderr)

    # ---------- run generation ----------
    results = []
    from vllm import LLM  # ensure namespace exists
    for i in tqdm(range(0, len(tasks), bsz), desc="Generating responses"):
        batch = tasks[i:i+bsz]
        outputs = llm.generate([t.prompt_text for t in batch], sampling_params=sampling_params, use_tqdm=False)
        for t, out in zip(batch, outputs):
            cand = out.outputs[0] if out.outputs else None
            text = cand.text if cand else ""

            # robust avg logp
            token_lps = getattr(cand, "token_logprobs", None) if cand else None
            if token_lps is None and cand is not None:
                lp_list = getattr(cand, "logprobs", None)
                tok_ids = getattr(cand, "token_ids", None) or getattr(cand, "output_token_ids", None)
                token_lps = []
                if lp_list is not None and tok_ids is not None and len(lp_list) == len(tok_ids):
                    for cands, tid in zip(lp_list, tok_ids):
                        lp = None
                        if isinstance(cands, dict):
                            entry = cands.get(tid) or cands.get(int(tid)) or cands.get(str(tid))
                            if entry is not None:
                                if hasattr(entry, "logprob"):
                                    try: lp = float(entry.logprob)
                                    except Exception: lp = None
                                elif isinstance(entry, (int, float)):
                                    lp = float(entry)
                        token_lps.append(lp)

            vals = [x for x in (token_lps or []) if x is not None and math.isfinite(x)]
            avg_logp = float(np.mean(vals)) if vals else float("nan")
            n_tok = len(vals)

            pred_raw = extract_final_answer(text)
            pred_norm = normalize_answer(pred_raw)

            gold_raw = gold_by_key.get(t.sample_key, "")
            gold_norm = normalize_answer(gold_raw)

            is_match = int(pred_norm == gold_norm) if gold_norm else None

            record = {
                "uuid": t.sample_key.uuid,
                "gen_index": t.sample_key.gen_index,
                "variant": t.variant,
                "step_index": t.step_index,
                "pred_text": text,
                "pred_extracted": pred_raw,
                "pred_norm": pred_norm,
                "gold_raw": gold_raw,
                "gold_norm": gold_norm,
                "is_match": is_match,
                "granularity": granularity,
                "model": str(model_name),
                "max_new_tokens": max_new_tokens,
                "gpu_mem_util": gpu_mem_util,
                "max_model_len": max_model_len,
                "offline": int(offline),
                "ctx_headroom": ctx_headroom,
                "avg_logp": avg_logp,
                "gen_n_tokens": n_tok
            }
            results.append(record)

    # ---------- importance aggregation ----------
    def _normalize_array(arr, method: str):
        x = np.array(arr, dtype=float)
        if method == "none":
            return x.tolist()
        if method == "minmax":
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
            if not math.isfinite(lo) or not math.isfinite(hi) or hi == lo:
                return [0.0 if math.isfinite(v) else 0.0 for v in x]
            return [float((v - lo) / (hi - lo)) if math.isfinite(v) else 0.0 for v in x]
        if method == "softmax":
            m = float(np.nanmax(x)) if np.isfinite(np.nanmax(x)) else 0.0
            exps = np.exp(np.where(np.isfinite(x), x - m, -1e9))
            s = float(np.sum(exps)) or 1.0
            return [float(v / s) for v in exps]
        if method == "zscore":
            mu = float(np.nanmean(x)) if np.isfinite(np.nanmean(x)) else 0.0
            sd = float(np.nanstd(x)); sd = sd if (np.isfinite(sd) and sd > 0) else 1.0
            return [float((v - mu) / sd) if math.isfinite(v) else 0.0 for v in x]
        raise ValueError(f"Unknown normalize method: {method}")

    # group by (uuid, gen_index)
    by_sid = defaultdict(lambda: {"base_lp": None, "abl_lp": {}, "base_acc": None, "abl_acc": {}})
    for r in results:
        sid = f"{r['uuid']}|{r['gen_index']}"
        if r["step_index"] == -1:
            by_sid[sid]["base_lp"] = r.get("avg_logp", float("nan"))
            by_sid[sid]["base_acc"] = r.get("is_match", None)
        else:
            by_sid[sid]["abl_lp"][r["step_index"]] = r.get("avg_logp", float("nan"))
            by_sid[sid]["abl_acc"][r["step_index"]] = r.get("is_match", None)

    # compute importance sequences
    imp_by_sid = {}
    for sid, pack in by_sid.items():
        base_lp = pack["base_lp"]
        K = (max(pack["abl_lp"].keys()) + 1) if pack["abl_lp"] else 0
        lp_seq = [pack["abl_lp"].get(i, float("nan")) for i in range(K)]

        raw_lp = []
        if isinstance(base_lp, (int, float)) and math.isfinite(base_lp):
            raw_lp = [max(0.0, base_lp - v) if (isinstance(v, (int, float)) and math.isfinite(v)) else 0.0
                      for v in lp_seq]
        imp_lp = _normalize_array(raw_lp, method=normalize) if raw_lp else []

        base_acc = pack["base_acc"]
        acc_seq = [pack["abl_acc"].get(i, None) for i in range(K)]
        raw_acc = []
        if base_acc is not None:
            raw_acc = [max(0, int(base_acc) - int(v)) if (v is not None) else 0 for v in acc_seq]  # 1->0 = drop
        # 0/1 本身即可作为重要性，不再归一
        imp_acc = raw_acc

        imp_by_sid[sid] = {"raw_lp": raw_lp, "imp_lp": imp_lp, "base_lp": base_lp,
                           "raw_acc": raw_acc, "imp_acc": imp_acc, "base_acc": base_acc}

    # fill back
    for r in results:
        sid = f"{r['uuid']}|{r['gen_index']}"
        pack = imp_by_sid.get(sid)
        if not pack:
            continue
        if r["step_index"] == -1:
            r["baseline_avg_logp"] = pack["base_lp"]
            r["baseline_is_match"] = pack["base_acc"]
        else:
            k = r["step_index"]
            if k < len(pack["imp_lp"]):
                r["logp_raw_delta"] = pack["raw_lp"][k]
                r["logp_importance"] = pack["imp_lp"][k]
            if k < len(pack["imp_acc"]):
                r["acc_raw_delta"] = pack["raw_acc"][k]
                r["acc_importance"] = pack["imp_acc"][k]

    write_jsonl(out_path, results)
    print(f"Wrote {len(results)} records to {out_path}", file=sys.stderr)

# --------------------
# CLI
# --------------------
def main():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
                        help="Prefer a local directory for offline runs.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "auto"])
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--offline", action="store_true", default=True)

    # data (two modes; summary first)
    parser.add_argument("--summary-jsonl", type=str, default="/root/autodl-fs/out_openr1_answer_summaries.jsonl",
                        help="If provided, the script reads from this file and segments steps on the fly.")
    # parser.add_argument("--raw-masked", type=str, default="/root/autodl-tmp/out200/summary/raw_masked.jsonl")
    parser.add_argument("--raw-masked", type=str, default = "/root/autodl-tmp/out2000/new_prompt/summary/raw_masked.jsonl")
    parser.add_argument("--macro-steps", type=str, default="/root/autodl-tmp/out2000/new_prompt/summary/macro_steps.jsonl")
    parser.add_argument("--sentences", type=str, default="/root/autodl-tmp/out200/new_prompt/summary/sentences.jsonl")

    # behavior
    parser.add_argument("--granularity", type=str, choices=["macro", "sentence"], default="sentence")
    parser.add_argument("--out", type=str, default="/root/autodl-fs/out2000/new_prompt/summary/math/ablation_from_summary.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--gpu-mem-util", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--download-dir", type=str, default=None)
    parser.add_argument("--ctx-headroom", type=int, default=64)
    parser.add_argument("--importance", type=str, default="logp", choices=["logp", "acc_drop"])
    parser.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "softmax", "zscore", "none"])
    parser.add_argument("--stop", type=str, nargs="*", default=None)
    args = parser.parse_args()

    run_ablation(
        model_name=args.model,
        granularity=args.granularity,
        out_path=args.out,
        summary_jsonl=args.summary_jsonl if args.summary_jsonl else None,
        raw_masked_path=args.raw_masked,
        macro_steps_path=args.macro_steps,
        sentences_path=args.sentences,
        max_new_tokens=args.max_new_tokens,
        bsz=args.bsz,
        limit=args.limit,
        dtype=args.dtype,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        download_dir=args.download_dir,
        offline=args.offline,
        ctx_headroom=args.ctx_headroom,
        importance=args.importance,
        normalize=args.normalize,
        stop=args.stop,
        tp=args.tp,
    )

if __name__ == "__main__":
    main()
