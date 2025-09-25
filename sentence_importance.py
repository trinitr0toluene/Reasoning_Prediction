# -*- coding: utf-8 -*-
"""
Sentence Importance Pipeline (ACC-Δ / LL-Δ) for OpenR1-Math style datasets.

This module assumes you already ran preprocessing to produce the following JSONL files:
- raw_masked.jsonl   : includes problem text, gold answer, masked reasoning, etc.
- sentences.jsonl    : sentence-level segmentation with sid, text per (uuid, gen_index)
- macro_steps.jsonl  : optional macro grouping (start_sid/end_sid), not required but useful.

Outputs per-sample JSON (see `evaluate_dataset`), with per-sentence I_out scores and optional I_proc.

How to use
----------
1) Implement a GenerationModel adapter (see classes below).
   - For ACC-Δ: implement `generate_once(ctx_text, seed)`
   - For LL-Δ  : implement `logprob_answer(ctx_text, answer_tokens)`

2) Provide a context template (prompt) via ContextBuilder (defaults included).

3) Run:
    from sentence_importance import run_cli
    # Or: python sentence_importance.py --help

Design choices
--------------
- Answer shielding: we assume your "raw_masked.jsonl" already masked gold answers with a token like ⟦ANS⟧.
- ACC-Δ is Monte Carlo accuracy drop after removing/replacing a sentence.
- LL-Δ is log-likelihood drop of the gold answer (tokenized) after removing/replacing a sentence.
- Counterfactual generation uses minimal rule-based edits (sign flip, inequality flip, small coefficient tweaks).
- All expensive calls are cached by a stable hash key under `cache_dir`.

Author: (you)
"""

from __future__ import annotations
import json, re, math, random, hashlib, time, os, sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable, Callable
from dataclasses import dataclass, field

# -------------------------
# Utils
# -------------------------

def md5_of(obj: Any) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False)+"\n")

# -------------------------
# Answer equivalence
# -------------------------

BOXED_RE = re.compile(r"\\boxed\{(.+?)\}")

def extract_boxed(text: str) -> Optional[str]:
    m = list(BOXED_RE.finditer(text))
    if not m: return None
    return f"\\boxed{{{m[-1].group(1).strip()}}}"

def _strip_boxed(a: str) -> str:
    m = BOXED_RE.fullmatch(a.strip())
    return m.group(1).strip() if m else a.strip()

def _normalize_spaces(s: str) -> str:
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _numeric_eval(s: str) -> Optional[float]:
    # Attempt to evaluate a simple numeric expression; fallback None.
    try:
        # Very conservative: only digits, parentheses, +-*/, fractions.
        if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", s):
            return None
        return float(eval(s, {"__builtins__":{}}))
    except Exception:
        return None

def answers_equiv(pred: str, gold: str, rtol=1e-6, atol=1e-8) -> bool:
    """Loose equivalence: exact boxed equality OR numeric near-equality for simple numbers."""
    if not pred or not gold: return False
    p, g = extract_boxed(pred) or pred, extract_boxed(gold) or gold
    p_core, g_core = _normalize_spaces(_strip_boxed(p)), _normalize_spaces(_strip_boxed(g))
    if p_core == g_core:
        return True
    # try numeric
    p_num, g_num = _numeric_eval(p_core), _numeric_eval(g_core)
    if p_num is not None and g_num is not None:
        return math.isclose(p_num, g_num, rel_tol=rtol, abs_tol=atol)
    return False

# -------------------------
# Model adapter interfaces
# -------------------------

class GenerationModel:
    """Abstract adapter for a generation model.
    Implement at least one of:
      - generate_once(ctx_text, seed) -> str
      - logprob_answer(ctx_text, answer_tokens) -> float
    """
    def generate_once(self, ctx_text: str, seed: Optional[int]=None) -> str:
        raise NotImplementedError

    def logprob_answer(self, ctx_text: str, answer_tokens: List[int]) -> float:
        raise NotImplementedError

    # Optional: tokenization utilities for LL-Δ
    def tokenize_answer(self, answer_text: str) -> List[int]:
        """Return token IDs for the answer_text. Override for LL-Δ mode."""
        raise NotImplementedError

class DummyEchoModel(GenerationModel):
    """A minimal dummy model for wiring test. DO NOT use for real evaluation."""
    def generate_once(self, ctx_text: str, seed: Optional[int]=None) -> str:
        # Just echo the last boxed in ctx if present; else emits a placeholder.
        last = extract_boxed(ctx_text)
        return last or r"\\boxed{0}"

# -------------------------
# Context building
# -------------------------

DEFAULT_INFER_PROMPT = """你将看到一道数学题的题面与若干已给出的推理步骤（其中 ⟦ANS⟧ 表示某未知常数）。
请仅基于这些内容给出最终答案，禁止输出任何思考或解释。
如果答案是一个数或代数表达式，请用 \\boxed{...} 的形式输出。

【题目】
{problem}

【已给推理（按顺序，含 sid）]
{sentences}

只输出一行 \\boxed{{...}}，不要额外文字。
"""

def render_sentences_for_ctx(sentences: List[Dict[str, Any]], sids_keep: Optional[set]=None, replacements: Optional[Dict[int,str]]=None) -> str:
    rows = []
    for s in sentences:
        sid = s["sid"]
        if sids_keep is not None and sid not in sids_keep:
            continue
        text = s["text"]
        if replacements and sid in replacements:
            text = replacements[sid]
        rows.append(f"[sid={sid}] {text}")
    return "\n".join(rows)

@dataclass
class ContextBuilder:
    template: str = DEFAULT_INFER_PROMPT

    def build(self, problem: str, sentences: List[Dict[str, Any]], sids_keep: Optional[set]=None, replacements: Optional[Dict[int,str]]=None) -> str:
        body = render_sentences_for_ctx(sentences, sids_keep=sids_keep, replacements=replacements)
        return self.template.format(problem=problem, sentences=body)

# -------------------------
# Counterfactual generator (heuristic)
# -------------------------

NUM_RE = re.compile(r"(?P<sign>[-+])?(?P<int>\d+)(?P<frac>/\d+)?")

def gen_counterfactual_variants(sent_text: str, max_k: int=3) -> List[str]:
    """Rule-based minimal edits: sign flip, small coefficient tweak, inequality flip."""
    outs = []
    t = sent_text

    # 1) inequality flip
    flips = {">=":"<=", "<=":">=", ">":"<", "<":">"}
    for a,b in flips.items():
        if a in t:
            outs.append(t.replace(a,b,1))
            break

    # 2) sign flip near a number (+x -> -x)
    m = re.search(r"([\+\-])\s*(\d+)", t)
    if m:
        start, end = m.span(1)
        sign = m.group(1)
        flipped = "-" if sign == "+" else "+"
        outs.append(t[:start] + flipped + t[start+1:])

    # 3) small coefficient tweak (n -> n±1)
    for m in NUM_RE.finditer(t):
        s, i, frac = m.group("sign") or "", m.group("int"), m.group("frac") or ""
        if frac: continue
        n = int(i)
        tweaked = f"{s}{max(0,n-1)}"
        outs.append(t[:m.start()] + tweaked + t[m.end():])
        if len(outs) >= max_k: break

    # de-dup and ensure not identical
    outs = [o for o in dict.fromkeys(outs) if o and o != sent_text]
    return outs[:max_k]

# -------------------------
# Caching layer
# -------------------------

@dataclass
class DiskCache:
    root: Path

    def __post_init__(self):
        ensure_dir(self.root)

    def get(self, key: Dict[str, Any]) -> Optional[Any]:
        h = md5_of(key)
        p = self.root / f"{h}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def put(self, key: Dict[str, Any], value: Any):
        h = md5_of(key)
        p = self.root / f"{h}.json"
        p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

# -------------------------
# ACC-Δ evaluator
# -------------------------

@dataclass
class AccDeltaConfig:
    N: int = 80
    temperature: float = 0.7
    top_p: float = 0.9
    seeds: Optional[List[int]] = None
    use_counterfactual: bool = True
    max_cf: int = 3

@dataclass
class AccDeltaEvaluator:
    model: GenerationModel
    ctx_builder: ContextBuilder
    cache: DiskCache
    cfg: AccDeltaConfig = field(default_factory=AccDeltaConfig)

    def _acc(self, ctx_text: str, answer_gt: str) -> float:
        seeds = self.cfg.seeds or list(range(self.cfg.N))
        correct = 0
        for s in seeds[:self.cfg.N]:
            out = self.model.generate_once(ctx_text, seed=s)
            if answers_equiv(out, answer_gt):
                correct += 1
        return correct / float(self.cfg.N)

    def acc_delta_for_sentence(self, problem: str, answer_gt: str, sentences: List[Dict[str, Any]], sid: int) -> Dict[str, Any]:
        key_full = {"kind":"acc_full","problem":problem,"sids":"ALL","ans":answer_gt}
        p_full = self.cache.get(key_full)
        if p_full is None:
            ctx_full = self.ctx_builder.build(problem, sentences)
            p_full = self._acc(ctx_full, answer_gt)
            self.cache.put(key_full, p_full)

        # delete
        sids_keep = {s["sid"] for s in sentences if s["sid"] != sid}
        key_del = {"kind":"acc_del","problem":problem,"sid":sid,"ans":answer_gt}
        p_del = self.cache.get(key_del)
        if p_del is None:
            ctx_del = self.ctx_builder.build(problem, sentences, sids_keep=sids_keep)
            p_del = self._acc(ctx_del, answer_gt)
            self.cache.put(key_del, p_del)

        delta_del = max(0.0, p_full - p_del)
        delta_cf = 0.0

        if self.cfg.use_counterfactual:
            # generate a few counterfactual variants
            sent_text = next(s["text"] for s in sentences if s["sid"] == sid)
            cfs = gen_counterfactual_variants(sent_text, max_k=self.cfg.max_cf) or []
            best_drop = 0.0
            for i, cf in enumerate(cfs):
                key_cf = {"kind":"acc_cf","problem":problem,"sid":sid,"cf_i":i,"cf":cf,"ans":answer_gt}
                p_cf = self.cache.get(key_cf)
                if p_cf is None:
                    ctx_cf = self.ctx_builder.build(problem, sentences, replacements={sid: cf})
                    p_cf = self._acc(ctx_cf, answer_gt)
                    self.cache.put(key_cf, p_cf)
                best_drop = max(best_drop, p_full - p_cf)
            delta_cf = max(0.0, best_drop)

        return {
            "sid": sid,
            "delta_del": round(delta_del, 6),
            "delta_cf": round(delta_cf, 6),
            "I_out": round(max(delta_del, delta_cf), 6),
            "N": self.cfg.N
        }

# -------------------------
# LL-Δ evaluator (optional, needs logprobs)
# -------------------------

@dataclass
class LlDeltaConfig:
    length_norm: bool = True

@dataclass
class LlDeltaEvaluator:
    model: GenerationModel
    ctx_builder: ContextBuilder
    cache: DiskCache
    cfg: LlDeltaConfig = field(default_factory=LlDeltaConfig)

    def _ll(self, ctx_text: str, answer_text: str) -> float:
        tokens = self.model.tokenize_answer(answer_text)
        ll = self.model.logprob_answer(ctx_text, tokens)  # sum of logprobs
        if self.cfg.length_norm and tokens:
            ll = ll / len(tokens)
        return ll

    def ll_delta_for_sentence(self, problem: str, answer_gt: str, sentences: List[Dict[str, Any]], sid: int) -> Dict[str, Any]:
        key_full = {"kind":"ll_full","problem":problem,"sids":"ALL","ans":answer_gt}
        ll_full = self.cache.get(key_full)
        if ll_full is None:
            ctx_full = self.ctx_builder.build(problem, sentences)
            ll_full = self._ll(ctx_full, answer_gt)
            self.cache.put(key_full, ll_full)

        sids_keep = {s["sid"] for s in sentences if s["sid"] != sid}
        key_del = {"kind":"ll_del","problem":problem,"sid":sid,"ans":answer_gt}
        ll_del = self.cache.get(key_del)
        if ll_del is None:
            ctx_del = self.ctx_builder.build(problem, sentences, sids_keep=sids_keep)
            ll_del = self._ll(ctx_del, answer_gt)
            self.cache.put(key_del, ll_del)

        delta_del = max(0.0, ll_full - ll_del)
        delta_cf = 0.0

        # CF
        sent_text = next(s["text"] for s in sentences if s["sid"] == sid)
        cfs = gen_counterfactual_variants(sent_text, max_k=3) or []
        best_drop = 0.0
        for i, cf in enumerate(cfs):
            key_cf = {"kind":"ll_cf","problem":problem,"sid":sid,"cf_i":i,"cf":cf,"ans":answer_gt}
            ll_cf = self.cache.get(key_cf)
            if ll_cf is None:
                ctx_cf = self.ctx_builder.build(problem, sentences, replacements={sid: cf})
                ll_cf = self._ll(ctx_cf, answer_gt)
                self.cache.put(key_cf, ll_cf)
            best_drop = max(best_drop, ll_full - ll_cf)
        delta_cf = max(0.0, best_drop)

        return {
            "sid": sid,
            "delta_del_ll": round(delta_del, 6),
            "delta_cf_ll": round(delta_cf, 6),
            "I_out_ll": round(max(delta_del, delta_cf), 6),
        }

# -------------------------
# Dataset loading & join
# -------------------------

@dataclass
class Sample:
    uuid: str
    gen_index: int
    problem: str
    answer_gold: str
    sentences: List[Dict[str, Any]]

def load_joined_samples(path_raw: Path, path_sent: Path, limit: Optional[int]=None) -> List[Sample]:
    raw_rows = load_jsonl(path_raw)
    sent_rows = load_jsonl(path_sent)
    # build index for sentences
    sent_index: Dict[Tuple[str,int], Dict[str, Any]] = {}
    for r in sent_rows:
        sent_index[(r["uuid"], r["gen_index"])] = r
    out: List[Sample] = []
    for r in raw_rows[:(limit or len(raw_rows))]:
        key = (r["uuid"], r["gen_index"])
        if key not in sent_index:
            continue
        srow = sent_index[key]
        out.append(Sample(
            uuid=r["uuid"],
            gen_index=int(r["gen_index"]),
            problem=r.get("problem") or r.get("prompt") or "",
            answer_gold=r.get("answer_gold") or r.get("gold_answer") or "",
            sentences=srow.get("sentences") or []
        ))
    return out

# -------------------------
# Driver
# -------------------------

def evaluate_dataset(
    model: GenerationModel,
    path_raw: Path,
    path_sent: Path,
    out_path: Path,
    cache_dir: Path,
    num_samples: Optional[int] = None,
    use_acc_delta: bool = True,
    use_ll_delta: bool = False,
    candidate_ratio: float = 0.4,
    random_seed: int = 42,
):
    random.seed(random_seed)
    ensure_dir(cache_dir)
    ensure_dir(out_path.parent)

    samples = load_joined_samples(path_raw, path_sent, limit=num_samples)
    ctx_builder = ContextBuilder()
    cache = DiskCache(cache_dir)

    acc_eval = AccDeltaEvaluator(model=model, ctx_builder=ctx_builder, cache=cache)
    ll_eval = LlDeltaEvaluator(model=model, ctx_builder=ctx_builder, cache=cache)

    results = []

    for s in samples:
        # candidate selection: pick last X% sentences + those with numbers/equations
        n = len(s.sentences)
        last_k = int(max(3, n * candidate_ratio))
        last_set = {sent["sid"] for sent in s.sentences[-last_k:]}
        has_num = {sent["sid"] for sent in s.sentences if re.search(r"\d|=|<|>|\\frac|\\times|\\cdot", sent["text"])}
        candidates = sorted(list(last_set | has_num))
        per_sent = []

        for sid in candidates:
            rec = {"sid": sid}
            if use_acc_delta:
                d = acc_eval.acc_delta_for_sentence(s.problem, s.answer_gold, s.sentences, sid)
                rec.update(d)
            if use_ll_delta:
                d2 = ll_eval.ll_delta_for_sentence(s.problem, s.answer_gold, s.sentences, sid)
                rec.update(d2)
            per_sent.append(rec)

        results.append({
            "uuid": s.uuid,
            "gen_index": s.gen_index,
            "answer_gt": s.answer_gold,
            "n_sentences": n,
            "candidates_evaluated": len(candidates),
            "sentences": s.sentences,
            "scores": sorted(per_sent, key=lambda x: x.get("I_out", x.get("I_out_ll", 0.0)), reverse=True)
        })

    write_jsonl(out_path, results)
    return out_path

# -------------------------
# CLI
# -------------------------

def run_cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="raw_masked.jsonl")
    ap.add_argument("--sentences", type=str, default="sentences.jsonl")
    ap.add_argument("--out", type=str, default="importance_scores.jsonl")
    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--no_acc", action="store_true")
    ap.add_argument("--ll", action="store_true")
    args = ap.parse_args()

    # TODO: replace DummyEchoModel with your real adapter
    model = DummyEchoModel()

    out_path = evaluate_dataset(
        model=model,
        path_raw=Path(args.raw),
        path_sent=Path(args.sentences),
        out_path=Path(args.out),
        cache_dir=Path(args.cache_dir),
        num_samples=args.num_samples,
        use_acc_delta= (not args.no_acc),
        use_ll_delta=args.ll,
    )
    print(f"Wrote results to: {out_path}")

if __name__ == "__main__":
    run_cli()
