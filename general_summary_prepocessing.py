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
    ap.add_argument("--input_jsonl", type=str, default = "/root/autodl-fs/out_glaive_answer_summaries.jsonl",
                    help="general 数据集的本地 JSONL，包含 summary 字段（或同义字段）")
    ap.add_argument("--outdir", type=str, default="/root/autodl-tmp/out2000/general/summary/out_general_prep",
                    help="输出目录，包含 raw.jsonl / sentences.jsonl")
    ap.add_argument("--tokenizer", type=str, default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
                    help="用于统计 token 数（可填本地 tokenizer 目录）")
    ap.add_argument("--limit", type=int, default=2000, help="仅处理前 N 条（调试）")
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
    return len(tok(s, add_special_tokens=False).input_ids)

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
        # 去掉开头编号/项目符号
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

    # 还原 LaTeX
    rough = [restore_latex(s, blocks) for s in rough]

    # 合并极短/口头语到后句
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

    # 生成带位置信息的句子结构
    sentences: List[Sentence] = []
    cursor = 0
    for idx, seg in enumerate(merged, start=1):
        pos = text.find(seg, cursor)
        if pos < 0:
            pos = cursor  # 容错：找不到时按顺序推进
        start, end = pos, pos + len(seg)
        tok_n = count_tokens(tok, seg)
        sentences.append(Sentence(idx, seg, start, end, tok_n))
        cursor = end
    return sentences

# ===================== Summary 抽取 =====================
def extract_summaries(ex: Dict[str, Any]) -> List[str]:
    """
    尽量通用地从记录里取“summary”：
    优先顺序：
      1) answer_summaries
      2) summary
      3) summaries
      4) summary_text / summary_en / summary_zh
    统一返回 list[str]
    """
    fields_priority = [
        "answer_summaries", "summary", "summaries",
        "summary_text", "summary_en", "summary_zh","summary_answer_driven"
    ]
    vals: List[str] = []
    for k in fields_priority:
        v = ex.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            vals.extend([normalize(str(x)) for x in v if str(x).strip()])
        else:
            s = normalize(str(v))
            if s:
                vals.append(s)
        if vals:
            break  # 命中一个字段就用它
    return vals

# ===================== 主流程 =====================
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # tokenizer
    tokenizer_id = args.tokenizer
    use_local = os.path.isdir(tokenizer_id)  # 传目录→本地
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
    with tqdm(desc="Processing summaries") as pbar:
        for idx, ex in enumerate(read_jsonl(args.input_jsonl)):
            summaries = extract_summaries(ex)
            if not summaries:
                continue

            # 支持一条样本多个 summary
            for gi, summ in enumerate(summaries):
                # 句子切分（不做 mask）
                sents = split_sentences(summ, tok)

                # 统计
                n_prompt = len(tok((ex.get("problem") or ex.get("question") or ex.get("prompt") or ""), add_special_tokens=False).input_ids)
                n_summ_tokens = len(tok(summ, add_special_tokens=False).input_ids)

                # 写 raw.jsonl
                rec_raw = {
                    "uuid": ex.get("uuid") or ex.get("id") or f"general-{idx}",
                    "gen_index": gi,
                    "problem": ex.get("problem") or ex.get("question") or ex.get("prompt") or "",
                    "summary_raw": summ,
                    "tokens_prompt": n_prompt,
                    "tokens_summary": n_summ_tokens,
                    "tokenizer": args.tokenizer,
                    # 透传可能有用的字段，避免丢信息
                    "source": ex.get("source"),
                    "meta": {k: ex.get(k) for k in ("dataset", "task_type") if k in ex}
                }
                fp_raw.write(json.dumps(rec_raw, ensure_ascii=False) + "\n")

                # 写 sentences.jsonl
                rec_sent = {
                    "uuid": rec_raw["uuid"],
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
