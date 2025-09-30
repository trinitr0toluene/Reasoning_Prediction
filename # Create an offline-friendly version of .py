
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional, Tuple

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

from dataclasses import dataclass

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

# --------------------
# Answer extraction & normalization
# --------------------

import re

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
ANS_TAG_RE = re.compile(r"(?:Final Answer|Answer)\s*[:：]\s*(.+)", re.IGNORECASE)

def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    # 1) \boxed{...}
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    # 2) "Final Answer: ..."
    m = ANS_TAG_RE.search(text)
    if m:
        # stop at first line break
        return m.group(1).strip().splitlines()[0].strip()
    # 3) Otherwise, take the last non-empty line
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text.strip()

LATEX_CLEAN_RE = re.compile(
    r"(\\mathrm\{[^}]*\}|\\text\{[^}]*\}|\\left|\\right|\\!|\\,|\\;|\\:|\\quad|\\qquad|\\hspace\{[^}]*\}|\\phantom\{[^}]*\})"
)
MATH_MARK_RE = re.compile(r"[\$\u200b]")  # remove $ and zero-width chars
WS_RE = re.compile(r"\s+")

def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = LATEX_CLEAN_RE.sub("", s)
    s = MATH_MARK_RE.sub("", s)
    # Remove enclosing parentheses if they wrap entire string
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    s = WS_RE.sub("", s)
    return s

# --------------------
# Prompt builder
# --------------------

SYSTEM_PROMPT = (
    "You are a precise math solver. Use the provided hints to compute the answer. "
    "Do NOT show steps. Respond with only the final answer, ideally in LaTeX like \\boxed{...}."
)

USER_TEMPLATE = """\
Solve the problem using the hints below. Do not explain your steps.

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
    # Apply the chat template for the target model
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# --------------------
# Data loading & alignment
# --------------------

def align_samples(
    raw_masked_path: str,
    macro_steps_path: str,
    sentences_path: str,
    granularity: str = "macro",
) -> List[Sample]:
    raw = read_jsonl(raw_masked_path)
    # Index raw by key
    raw_by_key: Dict[Key, dict] = {}
    for r in raw:
        raw_by_key[Key(uuid=r["uuid"], gen_index=int(r["gen_index"]))] = r

    # Choose segmentation source
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
            # Skip if the segmentation and raw don't align
            continue
        r = raw_by_key[key]
        steps = get_steps(s)
        full_trace = "\n".join(steps)
        samples.append(
            Sample(
                key=key,
                problem=r["problem"],
                gold_answer=r.get("answer_gold", ""),
                steps=steps,
                full_trace=full_trace,
            )
        )
    return samples

# --------------------
# Inference runner
# --------------------

@dataclass
class GenTask:
    # One forward pass request
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
    granularity: str,
    raw_masked_path: str,
    macro_steps_path: str,
    sentences_path: str,
    out_path: str,
    max_new_tokens: int = 32,
    bsz: int = 32,
    limit: Optional[int] = None,
    dtype: str = "float16",
    gpu_mem_util: float = 0.75,
    max_model_len: int = 8192,
    download_dir: Optional[str] = None,
    offline: bool = False,
    ctx_headroom: int = 64,   # [MOD] 可配置模板余量（默认 64）
):
    # --- Set env for offline & MKL threading (before importing heavy libs) ---
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Make MKL use GNU OpenMP (avoid libgomp conflict)
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Lazy imports after env setup
    from transformers import AutoTokenizer
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        print("ERROR: vLLM is required to run this script. Please install vllm.", file=sys.stderr)
        raise

    # Choose a default download/cache dir if not provided
    if download_dir is None:
        download_dir = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

    print(f"Loading tokenizer for {model_name} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=offline,   # <- critical for offline
    )

    print(f"Loading model {model_name} (dtype={dtype}) with vLLM ...", file=sys.stderr)
    llm = LLM(
        model=model_name,                 # strongly recommend a *local directory* when offline
        dtype=dtype,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        download_dir=download_dir,        # vLLM will only look here when offline
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=[],
    )

    print("Aligning samples ...", file=sys.stderr)
    samples = align_samples(
        raw_masked_path=raw_masked_path,
        macro_steps_path=macro_steps_path,
        sentences_path=sentences_path,
        granularity=granularity,
    )
    if limit is not None:
        samples = samples[:limit]
    print(f"Total aligned samples: {len(samples)}", file=sys.stderr)

    # ======================= [MOD] 头尾截断: 工具函数 =======================
    vtok = llm.get_tokenizer()  # 使用 vLLM 自己的 tokenizer，避免计数不一致

    def prompt_len(text: str) -> int:
        return len(vtok.encode(text))

    def fit_head_tail(problem: str, steps: List[str], keep_idx: int, budget: int, sep: str = "\n"):
        """
        [MOD] 给定步骤列表，在不超过 token 预算的前提下，尽量取“头 + 尾”并在中间放 <...>。
        - keep_idx = -1 表示 base（不删除任何步），否则表示 drop_k 时的“保留所有除 k 外的步骤”。
        - 返回: (hints_text, head_count, tail_count, final_tokens)
        """
        # 1) 选择参与拼接的 steps
        if keep_idx == -1:
            other = steps
        else:
            other = steps[:keep_idx] + steps[keep_idx+1:]

        # 2) 如果连空 hints 都放不下，则返回空字符串（上游再决定跳过）
        empty_prompt = build_chat_prompt(tokenizer, problem, "")
        if prompt_len(empty_prompt) > budget:
            return "", 0, 0, prompt_len(empty_prompt)

        # 3) 双端贪心：优先塞头部，再塞尾部，直到无法继续
        head, tail = [], []
        i, j = 0, len(other) - 1

        # helper to构建提示并计数
        def build_with(head_ls, tail_ls) -> Tuple[str, int]:
            hints = sep.join(head_ls + (["<...>"] if head_ls or tail_ls else []) + tail_ls)
            pr = build_chat_prompt(tokenizer, problem, hints)
            return hints, prompt_len(pr)

        # 先尽可能塞头（直观、简单；避免来回震荡）
        while i <= j:
            cand_head = head + [other[i]]
            hints, toks = build_with(cand_head, tail)
            if toks <= budget:
                head = cand_head
                i += 1
            else:
                break
        # 再尽可能塞尾
        while i <= j:
            cand_tail = [other[j]] + tail
            hints, toks = build_with(head, cand_tail)
            if toks <= budget:
                tail = cand_tail
                j -= 1
            else:
                break

        # 最终构建
        hints, toks = build_with(head, tail)
        return hints, len(head), len(tail), toks
    # ======================= [MOD END 工具函数] =======================

    # [MOD] 预算：为生成和模板留出余量
    CTX_BUDGET = max_model_len - max_new_tokens - ctx_headroom
    if CTX_BUDGET <= 0:
        raise ValueError(
            f"Invalid context budget: max_model_len({max_model_len}) - "
            f"max_new_tokens({max_new_tokens}) - headroom({ctx_headroom}) <= 0"
        )
    print(f"[truncate] context budget={CTX_BUDGET}, headroom={ctx_headroom}", file=sys.stderr)

    # [MOD] 记录被截断/无法容纳的样本
    trunc_logs: List[dict] = []
    impossible_logs: List[dict] = []

    # Build generation tasks  [MOD]：在构建时就做长度检查与截断
    tasks: List[GenTask] = []
    for s in samples:
        # base
        base_prompt = build_chat_prompt(tokenizer, s.problem, s.full_trace)
        base_len = prompt_len(base_prompt)
        base_hints = s.full_trace
        truncated_variants = []

        # 如果连空 hints 的 base 都放不下，记录为 impossible 并跳过该样本
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

        # drop_k
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

        # 若该样本有任何变体被截断，记录日志
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

    # [MOD] 写出截断/不可容纳日志
    if trunc_logs:
        trunc_path = os.path.splitext(out_path)[0] + ".truncated.jsonl"
        with open(trunc_path, "w", encoding="utf-8") as f:
            for rec in trunc_logs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[truncate] Wrote {len(trunc_logs)} truncated-sample records -> {trunc_path}", file=sys.stderr)

    if impossible_logs:
        imp_path = os.path.splitext(out_path)[0] + ".impossible.jsonl"
        with open(imp_path, "w", encoding="utf-8") as f:
            for rec in impossible_logs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[truncate] Wrote {len(impossible_logs)} impossible-sample records -> {imp_path}", file=sys.stderr)

    print(f"Total generation tasks (after truncation): {len(tasks)}", file=sys.stderr)

    # ---- 执行生成 ----
    from vllm import SamplingParams  # already imported above
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, stop=[]
    )

    results = []
    for i in range(0, len(tasks), bsz):
        batch = tasks[i:i+bsz]
        outputs = llm.generate([t.prompt_text for t in batch], sampling_params=sampling_params, use_tqdm=False)
        for t, out in zip(batch, outputs):
            text = out.outputs[0].text if out.outputs else ""
            pred_raw = extract_final_answer(text)
            pred_norm = normalize_answer(pred_raw)
            # gold 暂未用到
            record = {
                "uuid": t.sample_key.uuid,
                "gen_index": t.sample_key.gen_index,
                "variant": t.variant,
                "step_index": t.step_index,
                "pred_text": text,
                "pred_extracted": pred_raw,
                "pred_norm": pred_norm,
                "granularity": granularity,
                "model": str(model_name),
                "max_new_tokens": max_new_tokens,
                "gpu_mem_util": gpu_mem_util,
                "max_model_len": max_model_len,
                "offline": int(offline),
                "ctx_headroom": ctx_headroom,
            }
            results.append(record)

    write_jsonl(out_path, results)
    print(f"Wrote {len(results)} records to {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--granularity", type=str, choices=["macro", "sentence"], default="macro")
    parser.add_argument("--raw-masked", type=str, default="/mnt/data/raw_masked.jsonl")
    parser.add_argument("--macro-steps", type=str, default="/mnt/data/macro_steps.jsonl")
    parser.add_argument("--sentences", type=str, default="/mnt/data/sentences.jsonl")
    parser.add_argument("--out", type=str, default="/mnt/data/ablation_results.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "auto"])
    parser.add_argument("--gpu-mem-util", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--download-dir", type=str, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--ctx-headroom", type=int, default=64, help="[MOD] 头尾截断时为模板预留的 token 余量")

    args = parser.parse_args()

    run_ablation(
        model_name=args.model,
        granularity=args.granularity,
        raw_masked_path=args.raw_masked,
        macro_steps_path=args.macro_steps,
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
    )

if __name__ == "__main__":
    main()
