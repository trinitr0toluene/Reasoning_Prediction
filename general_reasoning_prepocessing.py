# -*- coding: utf-8 -*-
import argparse, json, os, unicodedata
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Iterable, Optional, Tuple
import regex
from tqdm import tqdm
from transformers import AutoTokenizer

# ===================== CLI =====================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_jsonl",
        type=str,
        default="/root/autodl-tmp/out2000/A+Q+RT/general/summary/ablation_sentence_summary.jsonl",
        help="包含 traces_natural 字段的 JSONL（例如 ablation_sentence_summary.jsonl）"
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="/root/autodl-tmp/out2000/A+Q+RT/general/trace_prep",
        help="输出目录，包含 raw.jsonl / sentences.jsonl"
    )
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
        help="用于统计 token 数（可填本地 tokenizer 目录）"
    )
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条（调试）")
    ap.add_argument(
        "--no_token_counts",
        action="store_true",
        help="不在预处理阶段统计 token 数，所有 tokens_* 字段与句子 tokens 置 0"
    )
    return ap.parse_args()

# ===================== 正则&工具 =====================
RE_LATEX_BLOCK = regex.compile(r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\))", flags=regex.S)
RE_BULLET = regex.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s+")
RE_SENT_END = regex.compile(r"([\.!?])(\s+|$)")
RE_META = regex.compile(r"^\s*(ok(ay)?|hmm+|let me|now|wait|note that|remember|i need to|i should|maybe|well|alright)\b", regex.I)

@dataclass
class Sentence:
    sid: int
    text: str
    start: int
    end: int
    tokens: int

def normalize(s: str) -> str:
    s = (s or "").replace("\r", "\n")
    s = unicodedata.normalize("NFKC", s)
    s = regex.sub(r"[ \t]+", " ", s)
    s = regex.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def count_tokens(tok, s: str) -> int:
    if tok is None:
        return 0
    return len(tok(s or "", add_special_tokens=False).input_ids)

def hold_latex_blocks(s: str) -> Tuple[str, List[str]]:
    blocks = []
    def repl(m):
        blocks.append(m.group(0))
        return f"⟦LATEX_{len(blocks)-1}⟧"
    return RE_LATEX_BLOCK.sub(repl, s), blocks

def restore_latex(s: str, blocks: List[str]) -> str:
    for i, b in enumerate(blocks):
        s = s.replace(f"⟦LATEX_{i}⟧", b)
    return s

def split_sentences(text_in: str, tok) -> List[Sentence]:
    """句子粒度切分：保护 LaTeX；按换行/编号/句末符号；合并极短口头语到后句；返回 char span + token 数。"""
    text = normalize(text_in)
    text_hold, blocks = hold_latex_blocks(text)

    rough: List[str] = []
    for line in text_hold.split("\n"):
        line = line.strip()
        if not line:
            continue
        if RE_BULLET.match(line):
            line = RE_BULLET.sub("", line)

        start = 0
        for m in RE_SENT_END.finditer(line):
            end = m.end(1)
            seg = line[start:end].strip()
            if seg:
                rough.append(seg)
            start = m.end()
        tail = line[start:].strip()
        if tail:
            rough.append(tail)

    rough = [restore_latex(s, blocks) for s in rough]

    merged: List[str] = []
    i = 0
    while i < len(rough):
        seg = rough[i].strip()
        if not seg:
            i += 1
            continue
        if (len(seg) <= 6 or RE_META.match(seg)) and i + 1 < len(rough):
            merged.append((seg + " " + rough[i+1]).strip())
            i += 2
        else:
            merged.append(seg)
            i += 1

    sentences: List[Sentence] = []
    cursor = 0
    for idx, seg in enumerate(merged, start=1):
        pos = text.find(seg, cursor)
        if pos < 0:
            pos = cursor
        start, end = pos, pos + len(seg)
        tok_n = count_tokens(tok, seg)
        sentences.append(Sentence(idx, seg, start, end, tok_n))
        cursor = end
    return sentences

# ===================== Trace 抽取 =====================
def extract_traces(ex: Dict[str, Any]) -> List[str]:
    """
    从记录里取 reasoning traces：
    优先使用 traces_natural（list[str] / str）。
    """
    vals: List[str] = []
    v = ex.get("traces_natural")
    if v is None:
        return vals
    if isinstance(v, list):
        for t in v:
            s = normalize(str(t))
            if s:
                vals.append(s)
    else:
        s = normalize(str(v))
        if s:
            vals.append(s)
    return vals

# ===================== 主流程 =====================
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # tokenizer（可选）
    tok = None
    if not args.no_token_counts:
        tokenizer_id = args.tokenizer
        use_local = os.path.isdir(tokenizer_id)
        tok = AutoTokenizer.from_pretrained(
            tokenizer_id,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=use_local
        )

    # 输出文件
    fp_raw = open(os.path.join(args.outdir, "raw.jsonl"), "w", encoding="utf-8")
    fp_sent = open(os.path.join(args.outdir, "sentences.jsonl"), "w", encoding="utf-8")

    n_rows = 0
    with tqdm(desc="Processing traces") as pbar:
        for idx, ex in enumerate(read_jsonl(args.input_jsonl)):
            traces = extract_traces(ex)
            if not traces:
                continue

            # 支持一条样本多个 trace（与 gen_index 对齐）
            for gi, trace in enumerate(traces):
                # 句子切分（对 trace）
                sents = split_sentences(trace, tok)

                # 统计（可关）
                problem_text = ex.get("problem") or ex.get("question") or ex.get("prompt") or ""
                n_prompt = count_tokens(tok, problem_text)
                n_trace_tokens = count_tokens(tok, trace)

                # 统一 id
                uid = ex.get("uuid") or ex.get("id") or ex.get("sample_id") or f"general-{idx}"

                # 写 raw.jsonl —— 保持兼容（summary_raw=trace），并补充 trace_raw
                rec_raw = {
                    "uuid": uid,
                    "gen_index": gi,
                    "problem": problem_text,
                    "final_answer": ex.get("final_answer", ""),
                    "summary_raw": trace,                # 兼容：下游若读取该字段
                    "trace_raw": trace,                  # 语义更清晰
                    "tokens_prompt": n_prompt,
                    "tokens_summary": n_trace_tokens,    # 兼容：值=trace token 数
                    "tokens_trace": n_trace_tokens,
                    "tokenizer": args.tokenizer if tok is not None else None,

                    # 透传原始信息
                    "traces_natural": traces,            # 原始 list
                    "traces_used_in_prompt": ex.get("traces_used_in_prompt", ""),
                    "response_raw": ex.get("response_raw", ""),
                    "source": ex.get("source"),
                    "meta": ex.get("meta", {}),
                }
                fp_raw.write(json.dumps(rec_raw, ensure_ascii=False) + "\n")

                # 写 sentences.jsonl（保持结构不变）
                rec_sent = {
                    "uuid": uid,
                    "gen_index": gi,
                    "n_sentences": len(sents),
                    "sentences": [asdict(s) for s in sents]
                }
                fp_sent.write(json.dumps(rec_sent, ensure_ascii=False) + "\n")

                n_rows += 1
                pbar.update(1)

                if args.limit and n_rows >= args.limit:
                    break
            if args.limit and n_rows >= args.limit:
                break

    fp_raw.close()
    fp_sent.close()
    print("Done. Files saved to", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
