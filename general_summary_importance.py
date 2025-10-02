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
    steps: List[str]         # sentence-level steps (来自 summary 的句子)
    full_trace: str          # 全部句子拼接

# --------------------
# Summary extraction & normalization
# --------------------
SUM_TAG_RE = re.compile(r"<summary>(.*?)</summary>", re.IGNORECASE | re.DOTALL)
WS_RE = re.compile(r"\s+")

def extract_summary(text: str) -> str:
    """优先抽取 <summary>...</summary>；否则用第一行兜底。"""
    if not text:
        return ""
    m = SUM_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else text.strip()

def normalize_summary(s: str) -> str:
    """摘要轻量标准化：合并空白，不做 LaTeX 清理。"""
    s = (s or "").strip()
    return WS_RE.sub(" ", s)

# --------------------
# Prompt builder (Summary)
# --------------------
SYSTEM_PROMPT = (
    "You are a precise summarizer. Read the problem and the provided hints. "
    "Write ONE concise, self-contained summary (<=40 words) of the reasoning leading to the answer. "
    "Do NOT show equations or detailed steps. Output ONLY <summary>...</summary>."
)

USER_TEMPLATE = """Summarize using ONLY the hints below.

[Problem]
{problem}

[Hints]
{hints}

Return exactly one line:
<summary>...your concise summary...</summary>
"""

def build_chat_prompt(tokenizer, problem: str, hints: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(problem=problem, hints=hints)},
    ]
    # 优先走模型自带 chat_template；没有则用朴素回退
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        sys_prompt = f"[SYSTEM]\n{SYSTEM_PROMPT}\n"
        usr_prompt = USER_TEMPLATE.format(problem=problem, hints=hints)
        return sys_prompt + "\n" + usr_prompt + "\n\nAssistant:"

# --------------------
# Data loading & alignment
# --------------------
def align_samples(
    raw_path: str,
    sentences_path: str,
) -> List[Sample]:
    raw = read_jsonl(raw_path)
    raw_by_key: Dict[Key, dict] = {}
    for r in raw:
        raw_by_key[Key(uuid=str(r["uuid"]), gen_index=int(r["gen_index"]))] = r

    seg = read_jsonl(sentences_path)
    samples: List[Sample] = []
    for s in seg:
        key = Key(uuid=str(s["uuid"]), gen_index=int(s["gen_index"]))
        if key not in raw_by_key:
            continue
        r = raw_by_key[key]
        problem = r.get("problem") or r.get("question") or r.get("prompt") or ""
        steps = [str(si["text"]) for si in s.get("sentences", []) if str(si.get("text","")).strip()]
        if not steps:
            continue
        full_trace = "\n".join(steps)
        samples.append(Sample(key=key, problem=problem, steps=steps, full_trace=full_trace))
    return samples

# --------------------
# Inference runner
# --------------------
@dataclass
class GenTask:
    sample_key: Key
    variant: str       # "base" or f"drop_{k}"
    step_index: int    # -1 for base; otherwise dropped step index
    prompt_text: str

def batched(iterable: Iterable, n: int) -> Iterable[List]:
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def run_ablation(
    model_name: str,
    raw_path: str,
    sentences_path: str,
    out_path: str,
    max_new_tokens: int = 48,
    bsz: int = 32,
    limit: Optional[int] = None,
    dtype: str = "float16",
    gpu_mem_util: float = 0.75,
    max_model_len: int = 8192,
    download_dir: Optional[str] = None,
    offline: bool = True,            # 默认离线
    ctx_headroom: int = 64,
    normalize: str = "minmax",
    tp: int = 1,
    stop: Optional[List[str]] = None,
):
    # --- Offline & threading env ---
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Lazy imports（在设置 env 之后）
    from transformers import AutoTokenizer
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        print("ERROR: vLLM is required to run this script. Please install vllm.", file=sys.stderr)
        raise

    if download_dir is None:
        download_dir = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

    print(f"Loading tokenizer for {model_name} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=True, local_files_only=offline,
    )

    print(f"Loading model {model_name} (dtype={dtype}) with vLLM ...", file=sys.stderr)
    llm = LLM(
        model=model_name, dtype=dtype, trust_remote_code=True,
        tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len, download_dir=download_dir,
    )

    # 固定停止符，稳定输出范围
    stop_list = list((stop or [])) + ["</summary>"]
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_new_tokens,
        stop=stop_list, logprobs=1,
    )

    print("Aligning samples ...", file=sys.stderr)
    samples = align_samples(raw_path=raw_path, sentences_path=sentences_path)
    if limit is not None:
        samples = samples[:limit]
    print(f"Total aligned samples: {len(samples)}", file=sys.stderr)

    # ---- Token budget helpers ----
    vtok = llm.get_tokenizer()
    def prompt_len(text: str) -> int:
        return len(vtok.encode(text))

    def fit_head_tail(problem: str, steps: List[str], drop_idx: int, budget: int, sep: str = "\n"):
        """在不超预算下构造 head+tail hints；drop_idx=-1 表示 base。"""
        kept = steps if drop_idx == -1 else (steps[:drop_idx] + steps[drop_idx+1:])

        empty_prompt = build_chat_prompt(tokenizer, problem, "")
        if prompt_len(empty_prompt) > budget:
            return "", 0, 0, prompt_len(empty_prompt)

        head, tail = [], []
        i, j = 0, len(kept) - 1

        def build_with(hh, tt) -> Tuple[str, int]:
            hints = sep.join(hh + (["<...>"] if (hh or tt) and (len(hh)+len(tt) < len(kept)) else []) + tt)
            pr = build_chat_prompt(tokenizer, problem, hints)
            return hints, prompt_len(pr)

        # 先尽量塞头
        while i <= j:
            cand_head = head + [kept[i]]
            hints, toks = build_with(cand_head, tail)
            if toks <= budget:
                head = cand_head; i += 1
            else:
                break
        # 再尽量塞尾
        while i <= j:
            cand_tail = [kept[j]] + tail
            hints, toks = build_with(head, cand_tail)
            if toks <= budget:
                tail = cand_tail; j -= 1
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

    # Build generation tasks
    tasks: List[GenTask] = []
    for s in samples:
        # Base
        base_hints = s.full_trace
        base_prompt = build_chat_prompt(tokenizer, s.problem, base_hints)
        base_len = prompt_len(base_prompt)

        empty_len = prompt_len(build_chat_prompt(tokenizer, s.problem, ""))
        if empty_len > CTX_BUDGET or empty_len > max_model_len:
            impossible_logs.append({
                "uuid": s.key.uuid, "gen_index": s.key.gen_index,
                "reason": "problem_too_long",
                "empty_prompt_tokens": empty_len,
                "ctx_budget": CTX_BUDGET, "max_model_len": max_model_len,
            })
            continue

        truncated_variants = []
        if base_len > CTX_BUDGET or base_len > max_model_len:
            ht, hcnt, tcnt, toks_after = fit_head_tail(s.problem, s.steps, drop_idx=-1, budget=CTX_BUDGET)
            base_hints = ht
            base_prompt = build_chat_prompt(tokenizer, s.problem, base_hints)
            truncated_variants.append({
                "variant": "base",
                "before_tokens": base_len, "after_tokens": toks_after,
                "head_steps": hcnt, "tail_steps": tcnt
            })
        tasks.append(GenTask(sample_key=s.key, variant="base", step_index=-1, prompt_text=base_prompt))

        # Per-sentence ablations
        for k in range(len(s.steps)):
            hints_k = "\n".join(s.steps[:k] + s.steps[k+1:])
            prompt_k = build_chat_prompt(tokenizer, s.problem, hints_k)
            len_k = prompt_len(prompt_k)
            if len_k > CTX_BUDGET or len_k > max_model_len:
                ht, hcnt, tcnt, toks_after = fit_head_tail(s.problem, s.steps, drop_idx=k, budget=CTX_BUDGET)
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
                "granularity": "sentence",
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

    # -------- Run generation in batches --------
    results = []
    for i in tqdm(range(0, len(tasks), bsz), desc="Generating summaries"):
        batch = tasks[i:i+bsz]
        outputs = llm.generate([t.prompt_text for t in batch], sampling_params=sampling_params, use_tqdm=False)
        for t, out in zip(batch, outputs):
            cand = out.outputs[0] if out.outputs else None
            text = cand.text if cand else ""

            # avg log-prob of generated tokens （兼容不同 vLLM 结构）
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
                                    try:
                                        lp = float(entry.logprob)
                                    except Exception:
                                        lp = None
                                elif isinstance(entry, (int, float)):
                                    lp = float(entry)
                        token_lps.append(lp)

            vals = [x for x in (token_lps or []) if x is not None and math.isfinite(x)]
            avg_logp = float(np.mean(vals)) if vals else float("nan")
            n_tok = len(vals)

            pred_sum = extract_summary(text)
            pred_norm = normalize_summary(pred_sum)

            record = {
                "uuid": t.sample_key.uuid,
                "gen_index": t.sample_key.gen_index,
                "variant": t.variant,
                "step_index": t.step_index,
                "pred_text": text,
                "pred_summary": pred_sum,
                "pred_summary_norm": pred_norm,
                "granularity": "sentence",
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

    # -------- Importance aggregation (logp drop) --------
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

    by_sid = defaultdict(lambda: {"base": None, "abl": {}})
    for r in results:
        sid = f"{r['uuid']}|{r['gen_index']}"
        if r["step_index"] == -1:
            by_sid[sid]["base"] = r.get("avg_logp", float("nan"))
        else:
            by_sid[sid]["abl"][r["step_index"]] = r.get("avg_logp", float("nan"))

    imp_by_sid = {}
    for sid, pack in by_sid.items():
        base = pack["base"]
        if not isinstance(base, (int, float)) or not math.isfinite(base):
            continue
        K = (max(pack["abl"].keys()) + 1) if pack["abl"] else 0
        lp_seq = [pack["abl"].get(i, float("nan")) for i in range(K)]
        raw = [max(0.0, base - v) if (isinstance(v, (int, float)) and math.isfinite(v)) else 0.0
               for v in lp_seq]
        imp = _normalize_array(raw, method=normalize)
        imp_by_sid[sid] = {"raw": raw, "imp": imp, "base": base}

    for r in results:
        sid = f"{r['uuid']}|{r['gen_index']}"
        pack = imp_by_sid.get(sid)
        if not pack:
            continue
        if r["step_index"] == -1:
            r["baseline_avg_logp"] = pack["base"]
        else:
            k = r["step_index"]
            if k < len(pack["imp"]):
                r["logp_raw_delta"] = pack["raw"][k]
                r["logp_importance"] = pack["imp"][k]

    write_jsonl(out_path, results)
    print(f"Wrote {len(results)} records to {out_path}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
                        help="本地模型目录（离线模式下必须是本地）")
    # 默认指向 general/summary 预处理产物
    parser.add_argument("--raw", type=str, default="/root/autodl-tmp/out2000/general/summary/out_general_prep/raw.jsonl",
                        help="summary 预处理产物 raw.jsonl")
    parser.add_argument("--sentences", type=str, default="/root/autodl-tmp/out2000/general/summary/out_general_prep/sentences.jsonl",
                        help="summary 预处理产物 sentences.jsonl")
    parser.add_argument("--out", type=str, default="/root/autodl-tmp/out2000/general/summary/ablation_sentence_summary.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "auto"])
    parser.add_argument("--gpu-mem-util", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--download-dir", type=str, default=None)
    parser.add_argument("--offline", action="store_true", default=True,
                        help="强制离线（设置 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE）")
    parser.add_argument("--ctx-headroom", type=int, default=64, help="模板预留 token 余量")
    parser.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "softmax", "zscore", "none"])
    parser.add_argument("--tp", type=int, default=1, help="vLLM tensor_parallel_size")
    parser.add_argument("--stop", type=str, nargs="*", default=None, help="可选停止词列表（会自动追加 </summary>）")
    args = parser.parse_args()

    run_ablation(
        model_name=args.model,
        raw_path=args.raw,
        sentences_path=args.sentences,
        out_path=args.out,
        max_new_tokens=args.max_new_tokens,
        bsz=args.bsz,
        limit=args.limit,
        dtype=args.dtype,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        download_dir=args.download_dir,
        offline=args.offline,
        ctx_headroom=args.ctx_headroom,
        normalize=args.normalize,
        tp=args.tp,
        stop=args.stop,
    )

if __name__ == "__main__":
    main()
