# -*- coding: utf-8 -*-
import argparse, json, re, unicodedata, os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from math import inf

import regex
from unidecode import unidecode
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import islice
from tqdm import tqdm  # 进度条

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="default", choices=["default","extended","all"])
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--topk", type=int, default=0, help="为宏步骤预筛的个数（0=不筛）")
    ap.add_argument("--outdir", type=str, default="out")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前N条（调试）")
    return ap.parse_args()

# ---------------- 正则/模式 ----------------
RE_THINK = regex.compile(r"<think>\s*(.*?)\s*</think>", flags=regex.S|regex.I)
RE_LATEX_BLOCK = regex.compile(r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\))", flags=regex.S)
RE_BULLET = regex.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s+")
RE_SENT_END = regex.compile(r"([\.!?])(\s+|$)")
RE_EQUATION = regex.compile(r"(=|\\Rightarrow|≥|≤|!=|\\approx|\\sim|\\frac|\\cdot|\\times|\\div)")
# RE_CONCLUDE = regex.compile(r"\b(therefore|thus|hence|so|consequently|final answer|answer is|boxed)\b", regex.I)
RE_CONCLUDE = re.compile(
    r"\b(therefore|thus|hence|consequently|in conclusion|final answer|answer is)\b",
    re.I
)

RE_VERIFY = regex.compile(r"\b(verify|check|plug\s+in|substitute\s+back|substitution)\b", regex.I)
RE_META = regex.compile(r"^\s*(ok(ay)?|hmm+|let me|now|wait|note that|remember|i need to|i should|maybe|well|alright)\b", regex.I)
RE_SYMBOLIC_DENSE = regex.compile(r"[=+\-*/^]|\\frac|\\sqrt|\\sum|\\int|\\prod|\\lim|\\binom|\\alpha|\\beta|\\gamma|\\pi|\\theta")
RE_BOXED = regex.compile(r"\\boxed\{(.*?)\}")
RE_CONT = regex.compile(
    r"^\s*(simplify|which simplifies|combine like terms|so now|now we have|therefore we can write|thus)\b",
    regex.I
)


ROLE_PATTERNS = {
    "plan": regex.compile(r"\b(plan|strategy|let'?s|we (?:will|shall)|first|next|then|step|approach|assume)\b", regex.I),
    "introduce": regex.compile(r"\b(let|denote|define|set)\b", regex.I),
    "derive": RE_EQUATION,
    "substitute": regex.compile(r"\b(substitute|plug)\b", regex.I),
    "intermediate": regex.compile(r"\b(obtain|get|compute|simplif(?:y|ies)|combine|rearrange|collect|expand)\b", regex.I),
    "check": RE_VERIFY,
    "conclude": RE_CONCLUDE,
}

N = 20000
# ---------------- 数据结构 ----------------
@dataclass
class Sentence:
    sid: int
    text: str
    start: int
    end: int
    tokens: int

@dataclass
class MacroStep:
    mid: int
    start_sid: int
    end_sid: int
    text: str
    role: str
    has_equation: bool
    is_check: bool
    is_conclusion: bool
    tokens: int
    features: Dict[str, Any]

# ---------------- 工具函数 ----------------
def normalize(s: str) -> str:
    s = s.replace("\r","\n")
    s = unicodedata.normalize("NFKC", s)
    s = regex.sub(r"[ \t]+"," ", s)
    s = regex.sub(r"\n{3,}","\n\n", s)
    return s.strip()

def strip_think(text: str) -> str:
    m = RE_THINK.search(text)
    return m.group(1).strip() if m else text

def extract_reasonings(example: Dict[str,Any]) -> List[str]:
    """优先从 generations[*].content 的 <think>…</think> 取；否则回退到 solution。"""
    outs = []
    gens = example.get("generations")
    if isinstance(gens, list) and gens:
        for g in gens:
            if isinstance(g, dict):
                content = g.get("content") or ""
            else:
                content = str(g)
            think = strip_think(content)
            if think.strip():
                outs.append(normalize(think))
    if not outs:
        sol = (example.get("solution") or "").strip()
        if sol:
            outs = [normalize(sol)]
    return outs

def extract_model_answer(text: str) -> Optional[str]:
    """解析 reasoning 中的最后一个 \\boxed{...} 作为模型给出的答案。"""
    m = list(RE_BOXED.finditer(text))
    if not m: 
        return None
    return f"\\boxed{{{m[-1].group(1).strip()}}}"

def answer_core(a: Optional[str]) -> Optional[str]:
    if not a: return None
    a = a.strip()
    a = regex.sub(r"^\\boxed\{(.*)\}$", r"\1", a).strip()
    return a

def mask_answer(text: str, gold: str) -> str:
    """Answer Shield：遮蔽与 gold 等价的显式片段。"""
    if not gold: return text
    a_core = regex.sub(r"^\\boxed\{(.*)\}$", r"\1", gold).strip()
    esc = regex.escape(a_core)
    out = regex.sub(rf"\\boxed\{{\s*{esc}\s*\}}", "⟦ANS⟧", text)
    out = regex.sub(rf"(?<!\w){esc}(?!\w)", "⟦ANS⟧", out)
    out = regex.sub(rf"(=|\s≈\s)\s*{esc}", r"\1 ⟦ANS⟧", out)
    return out

def hold_latex_blocks(s: str) -> Tuple[str,List[str]]:
    blocks=[]
    def repl(m):
        blocks.append(m.group(0))
        return f"⟦LATEX_{len(blocks)-1}⟧"
    return RE_LATEX_BLOCK.sub(repl, s), blocks

def restore_latex(s: str, blocks: List[str]) -> str:
    for i,b in enumerate(blocks):
        s = s.replace(f"⟦LATEX_{i}⟧", b)
    return s

def count_tokens(tok, s: str) -> int:
    return len(tok(s, add_special_tokens=False).input_ids)

def split_sentences(reasoning_masked: str, tok) -> List[Sentence]:
    """分句：保护 LaTeX；按换行/编号/句末符切；合并极短元话语并入后句；给出 char span + token 数。"""
    text = normalize(reasoning_masked)
    text_hold, blocks = hold_latex_blocks(text)

    rough=[]
    # 粗按行
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

    # 还原 LaTeX
    rough=[restore_latex(s, blocks) for s in rough]

    # 合并极短/元话语 → 并入后句
    merged=[]
    i=0
    while i<len(rough):
        seg=rough[i].strip()
        if not seg: i+=1; continue
        if (len(seg)<=6 or RE_META.match(seg)) and i+1 < len(rough):
            merged.append((seg + " " + rough[i+1]).strip())
            i += 2
        else:
            merged.append(seg); i += 1

    # 生成 char spans（顺序匹配）
    sentences=[]
    cursor=0
    for idx, seg in enumerate(merged, start=1):
        pos = text.find(seg, cursor)
        if pos < 0:
            # 回退：直接顺延
            pos = cursor
        start, end = pos, pos+len(seg)
        tok_n = count_tokens(tok, seg)
        sentences.append(Sentence(idx, seg, start, end, tok_n))
        cursor = end
    return sentences

def guess_role(text: str) -> str:
    for role, pat in ROLE_PATTERNS.items():
        if pat.search(text):
            return role
    if RE_SYMBOLIC_DENSE.search(text): return "derive"
    return "other"

def cheap_features(mid:int, total:int, text:str, ans_core:str) -> Dict[str,Any]:
    pos = mid / max(1,total)
    feats = {
        "rel_pos": round(pos,3),
        "len_chars": len(text),
        "contains_boxed": int("\\boxed" in text),
        "support_hint": int(bool(ans_core and ans_core in text))
    }
    score = 0.2*pos + 0.15*min(1,len(text)/120) + 0.1*feats["contains_boxed"]
    feats["cheap_score"] = round(min(1.0, score),3)
    return feats

def macro_chunk(sentences: List[Sentence], tok) -> List[MacroStep]:
    chunks=[]
    buf: List[Sentence]=[]

    def flush():
        nonlocal buf, chunks
        if not buf: return
        text=" ".join(s.text for s in buf).strip()
        role = guess_role(text)
        has_eq = bool(RE_EQUATION.search(text) or RE_SYMBOLIC_DENSE.search(text))
        is_check = bool(RE_VERIFY.search(text))
        # is_conc = bool(RE_CONCLUDE.search(text) or "\\boxed" in text)
        # 结论判定时再额外加：或包含 \boxed
        is_conc = bool(RE_CONCLUDE.search(text) or "\\boxed" in text)
        tokens = count_tokens(tok, text)
        feats = cheap_features(len(chunks)+1, 1, text, None)  # rel_pos 占位，稍后重算
        step = MacroStep(
            mid=len(chunks)+1,
            start_sid=buf[0].sid,
            end_sid=buf[-1].sid,
            text=text,
            role=role,
            has_equation=has_eq,
            is_check=is_check,
            is_conclusion=is_conc,
            tokens=tokens,
            features=feats
        )
        chunks.append(step); buf=[]

    i=0; n=len(sentences)
    while i<n:
        s=sentences[i]; t=s.text

        # 元话语并入后一句
        if RE_META.match(t) and i+1<n:
            buf.append(s); buf.append(sentences[i+1]); i+=2; flush(); continue

        # 等式/代数化简连续合并
        if RE_EQUATION.search(t) or RE_SYMBOLIC_DENSE.search(t):
            buf.append(s); j=i+1
            while j<n and (RE_EQUATION.search(sentences[j].text) or RE_SYMBOLIC_DENSE.search(sentences[j].text)):
                buf.append(sentences[j]); j+=1
            i=j; flush(); continue

        # 验证段合并
        if RE_VERIFY.search(t):
            buf.append(s); take=min(3, n-(i+1))
            for k in range(take): buf.append(sentences[i+1+k])
            i += 1+take; flush(); continue
        
        # 过渡句并入后句
        if RE_CONT.match(t) and i+1 < n:
            buf.append(s); buf.append(sentences[i+1]); i += 2; flush(); continue

        # 结论类合并
        if RE_CONCLUDE.search(t) or "\\boxed" in t:
            buf.append(s)
            if i+1<n and (RE_CONCLUDE.search(sentences[i+1].text) or RE_VERIFY.search(sentences[i+1].text)):
                buf.append(sentences[i+1]); i+=2
            else:
                i+=1
            flush(); continue

        # 默认：单句一步
        buf.append(s); i+=1; flush()

    # 重算 rel_pos/cheap_score（依赖 n）
    total=len(chunks)
    for m in chunks:
        m.features = cheap_features(m.mid, total, m.text, None)
    return chunks

def pick_correctness(ex: Dict[str,Any], gi:int, model_ans:str, gold:str) -> Tuple[Optional[bool], str]:
    mv = ex.get("correctness_math_verify")
    if isinstance(mv, list) and gi < len(mv) and mv[gi] is not None:
        return bool(mv[gi]), "math_verify"
    lj = ex.get("correctness_llama")
    if isinstance(lj, list) and gi < len(lj) and lj[gi] is not None:
        return bool(lj[gi]), "llama_judge"
    # 回退：字符串等价（仅作兜底，不做严格数值等价）
    if model_ans and gold:
        return (answer_core(model_ans)==answer_core(gold)), "string_match"
    return None, "unknown"


def _count_tokens(text: str, tok=None) -> int:
    if tok is None:
        return len(text.split())  # 兜底：空格分词
    try:
        return len(tok(text, add_special_tokens=False).input_ids)
    except Exception:
        return len(text.split())

def _is_hard_boundary(a, b) -> bool:
    """
    永不跨越的边界：结论步、验算步、从引入/计划直接切到推导的首个等式。
    """
    if getattr(a, "is_conclusion", False) or getattr(b, "is_conclusion", False):
        return True
    if getattr(a, "is_check", False) or getattr(b, "is_check", False):
        return True
    # 引入/计划 -> 推导 的转折（避免吞掉“开始推导”的第一步）
    if (a.role in {"introduce", "plan"}) and (b.role in {"derive", "intermediate"}) and (b.has_equation):
        return True
    return False

def _boundary_cost(a, b) -> float:
    """
    合并代价：越小越优先合并。硬边界直接 +inf。
    设计原则：优先吞掉短块、口癖/过渡句；避免跨大角色/等式变化的边界。
    """
    if _is_hard_boundary(a, b):
        return inf

    cost = 0.0

    # 角色变化的惩罚（derive 与 conclude/check/plan 之间的“跨域”更大）
    if a.role != b.role:
        if (a.role in {"derive", "intermediate"}) ^ (b.role in {"derive", "intermediate"}):
            cost += 0.9
        else:
            cost += 0.4

    # 等式有无不一致：不鼓励合并
    if bool(a.has_equation) != bool(b.has_equation):
        cost += 0.8

    # 短块奖励合并（任一非常短）
    if min(getattr(a, "tokens", 0), getattr(b, "tokens", 0)) < 18:
        cost -= 0.8

    # 两侧字符都很短，再给点奖励
    if len(a.text) < 60 and len(b.text) < 60:
        cost -= 0.4

    # 如果后一句是过渡/口癖句，进一步鼓励合并
    if RE_CONT.match(b.text):
        cost -= 0.3

    # 长度差越大越不合并
    la, lb = max(1, len(a.text)), max(1, len(b.text))
    gap = abs(la - lb) / max(la, lb)
    cost += 0.2 * gap

    return cost

def _merge_pair(a, b, tok=None):
    """把相邻两块合并到 a 上，并更新关键字段。"""
    a.text = (a.text + " " + b.text).strip()
    a.end_sid = b.end_sid
    a.tokens = _count_tokens(a.text, tok)
    a.has_equation = bool(a.has_equation or b.has_equation)
    a.is_check = bool(a.is_check or b.is_check)
    a.is_conclusion = bool(a.is_conclusion or b.is_conclusion)
    # role：尽量保留非 'other' 的有信息角色
    if a.role == "other" and b.role != "other":
        a.role = b.role
    # features 简化更新（够用即可）
    feats = getattr(a, "features", {}) or {}
    feats["len_chars"] = len(a.text)
    feats["contains_boxed"] = int("\\boxed" in a.text)
    a.features = feats
    return a

def _pack_by_token_budget(macros, tok=None, max_tokens=120):
    """保险丝：按 token 上限顺序打包，同时尊重硬边界。"""
    if not macros:
        return macros
    new = []
    buf = macros[0]
    for i in range(1, len(macros)):
        cur = macros[i]
        # 硬边界或打包超限：先收束，再开新块
        if _is_hard_boundary(buf, cur) or (_count_tokens(buf.text, tok) + _count_tokens(cur.text, tok) > max_tokens):
            new.append(buf)
            buf = cur
        else:
            buf = _merge_pair(buf, cur, tok)
    new.append(buf)
    return new

def _normalize_conclusion(macros):
    """把含 \\boxed 的步强制视为结论步；弱结论仅保留在尾部。"""
    n = max(1, len(macros))
    for i, m in enumerate(macros, 1):
        if "\\boxed" in m.text:
            m.is_conclusion = True
            m.role = "conclude"
        else:
            # 弱结论：仅尾部 30% 且命中结论词才标记
            if (i / n) >= 0.7 and RE_CONCLUDE.search(m.text or ""):
                m.is_conclusion = True
    return macros

def compress_macros(macros, tok=None, target_min=18, target_max=26):
    """
    在不引入新依赖的前提下，把步数压进 [target_min, target_max]。
    做法：先标准化结论标签 -> 贪心合并 -> 若仍超标，走 token 预算打包。
    """
    if not macros:
        return macros

    # 1) 结论标准化（强/弱结论）
    macros = _normalize_conclusion(macros)

    # 2) 贪心预算合并
    #    每轮选取“代价最低”的相邻边界进行合并，直到步数落入目标区间
    #    （或已无可合并边界）
    def _relabel_relpos(lst):
        n = max(1, len(lst))
        for i, m in enumerate(lst, 1):
            m.mid = i
            feats = getattr(m, "features", {}) or {}
            feats["rel_pos"] = round(i / n, 3)
            m.features = feats

    _relabel_relpos(macros)

    while len(macros) > target_max:
        best_i, best_cost = None, inf
        # 扫一遍所有相邻边界
        for i in range(len(macros) - 1):
            c = _boundary_cost(macros[i], macros[i + 1])
            if c < best_cost:
                best_cost, best_i = c, i
        # 没有可合并的边界（都为硬边界），跳保险丝
        if best_i is None or best_cost == inf:
            break
        # 合并最佳对
        merged = _merge_pair(macros[best_i], macros[best_i + 1], tok)
        macros[best_i] = merged
        macros.pop(best_i + 1)

    # 3) 保险丝：如果仍超标，用 token 预算顺序打包（尊重硬边界）
    if len(macros) > target_max:
        macros = _pack_by_token_budget(macros, tok, max_tokens=120)

    # 4) 最终重排 mid / rel_pos
    _relabel_relpos(macros)
    return macros

# ---------------- 主流程 ----------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading dataset open-r1/OpenR1-Math-220k ({args.split}) ...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", args.split, split="train")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # 展开：一题→多轨迹
    rows=[]
    for ex in tqdm(islice(ds, N), total=N, desc="Expanding problems → traces"):
        reasonings = extract_reasonings(ex)
        for gi, r in enumerate(reasonings):
            rows.append({
                "uuid": ex.get("uuid"),
                "problem": ex.get("problem"),
                "solution_human": ex.get("solution") or "",
                "answer_gold": ex.get("answer") or "",
                "source": ex.get("source"),
                "generations_len": len(reasonings),
                "gen_index": gi,
                "finish_reasons": (ex.get("finish_reasons") or [None]*len(reasonings))
            })
            rows[-1]["reasoning_raw"] = r

    if args.limit>0: rows = rows[:args.limit]
    print(f"Expanded to {len(rows)} problem-traces")

    # 三文件 writers
    fp_raw = open(os.path.join(args.outdir, "raw_masked.jsonl"), "w", encoding="utf-8")
    fp_sent = open(os.path.join(args.outdir, "sentences.jsonl"), "w", encoding="utf-8")
    fp_mac = open(os.path.join(args.outdir, "macro_steps.jsonl"), "w", encoding="utf-8")

    for ex in tqdm(rows, total=len(rows), desc="Preprocessing & writing"):
        # 解析模型答案 + 正确性
        model_ans = extract_model_answer(ex["reasoning_raw"])
        is_corr, verified_by = pick_correctness(ex, ex["gen_index"], model_ans, ex["answer_gold"])

        # 遮蔽答案
        masked = mask_answer(ex["reasoning_raw"], ex["answer_gold"])

        # token 计数
        n_prompt = count_tokens(tok, ex["problem"] or "")
        n_r_raw = count_tokens(tok, ex["reasoning_raw"])
        n_r_mask = count_tokens(tok, masked)
        n_ans_model = count_tokens(tok, model_ans or "")
        n_sol_human = count_tokens(tok, ex["solution_human"] or "")

        # 分句
        sents = split_sentences(masked, tok)

        # 宏步骤
        macros = macro_chunk(sents, tok)
        macros = compress_macros(macros, tok, target_min=18, target_max=26)

        # 补全 features 中对答案的支持提示（用 gold 核心作 hint）
        gold_core = answer_core(ex["answer_gold"])
        for m in macros:
            feats = m.features
            feats["rel_pos"] = round(m.mid / max(1,len(macros)), 3)
            feats["contains_boxed"] = int("\\boxed" in m.text)
            feats["support_hint"] = int(bool(gold_core and gold_core in m.text))
            m.features = feats

        # 预筛候选
        candidate_macro_ids=[]
        if args.topk and args.topk>0:
            sorted_ms = sorted(macros, key=lambda m: m.features.get("cheap_score",0), reverse=True)
            candidate_macro_ids = [m.mid for m in sorted_ms[:args.topk]]

        # ---- 写 raw_masked.jsonl ----
        rec_raw = {
            "uuid": ex["uuid"],
            "gen_index": ex["gen_index"],
            "problem": ex["problem"],
            "answer_gold": ex["answer_gold"],
            "solution_human": ex["solution_human"],
            "reasoning_raw": ex["reasoning_raw"],
            "reasoning_masked": masked,
            "model_answer": model_ans,
            "is_correct": is_corr,
            "verified_by": verified_by,
            "tokens_prompt": n_prompt,
            "tokens_reasoning_raw": n_r_raw,
            "tokens_reasoning_masked": n_r_mask,
            "tokens_model_answer": n_ans_model,
            "tokens_solution_human": n_sol_human,
            "tokenizer": args.tokenizer,
            "finish_reason": (ex["finish_reasons"][ex["gen_index"]] 
                              if isinstance(ex["finish_reasons"], list) and ex["gen_index"]<len(ex["finish_reasons"]) else None)
        }
        fp_raw.write(json.dumps(rec_raw, ensure_ascii=False) + "\n")

        # ---- 写 sentences.jsonl ----
        rec_sent = {
            "uuid": ex["uuid"],
            "gen_index": ex["gen_index"],
            "n_sentences": len(sents),
            "sentences": [asdict(s) for s in sents]
        }
        fp_sent.write(json.dumps(rec_sent, ensure_ascii=False) + "\n")

        # ---- 写 macro_steps.jsonl ----
        rec_mac = {
            "uuid": ex["uuid"],
            "gen_index": ex["gen_index"],
            "n_macros": len(macros),
            "macro_steps": [asdict(m) for m in macros],
            "candidate_macro_ids": candidate_macro_ids
        }
        fp_mac.write(json.dumps(rec_mac, ensure_ascii=False) + "\n")

    for fp in (fp_raw, fp_sent, fp_mac):
        fp.close()
    print("Done. Files saved to", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
