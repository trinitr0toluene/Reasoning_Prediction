# -*- coding: utf-8 -*-
"""
Math 数据集：关键覆盖与错失 (WCov / Miss@Top-q) + 词面一致性 + 幻觉粗检
----------------------------------------------------------------------------
输入（默认指向你给的三份 math 文件，可改命令行参数）：
  --summary_table /mnt/data/out_openr1_answer_summaries.jsonl
  --summary_ablation /mnt/data/ablation_from_summary.jsonl
  --reasoning_ablation /mnt/data/ablation_results.jsonl

核心做法：
  1) 从 summary_ablation（granularity=='sentence'）聚合出 wS（句级重要性）。
  2) 从 reasoning_ablation（granularity=='macro'）聚合出 wT（宏步重要性）。
  3) 用 out_openr1_answer_summaries：
     - S 文本：summary_answer_driven  -> 句子级分句
     - T 文本：traces_natural[gen_index] -> 按宏步数 K 等长切片得到 K 段文本（与 wT 维度一致）
  4) 计算相似度矩阵（TF-IDF→余弦；若 sklearn 不在则退化到 Jaccard），软对齐 A
  5) 指标：WCov / WCov@Top-q / Miss@Top-q、Jaccard/ROUGE-1/2/L、Unmatched_Ratio/Extraneous_Salience
  6) 仅对三个文件 **uuid & gen_index** 的交集样本计算，输出 CSV

用法示例：
python coverage_math.py \
  --summary_table /mnt/data/out_openr1_answer_summaries.jsonl \
  --summary_ablation /mnt/data/ablation_from_summary.jsonl \
  --reasoning_ablation /mnt/data/ablation_results.jsonl \
  --out_csv /mnt/data/math_coverage_results.csv \
  --topq 0.2 --theta 0.05 --tau 0.25
"""

import argparse, json, os, re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

# -------- 相似度（TF-IDF优先，失败退化到Jaccard） --------
USE_SK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    USE_SK = False

# -------- 基本读写 --------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# -------- 分句/分段 --------
_SENT_RE = re.compile(r"[。！？!?；;：:\.\n]+")

def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    return parts if parts else ([text.strip()] if text else [])

def equal_chunks(text: str, k: int) -> List[str]:
    """把长文本按字符等长切 K 段（近似宏步段）。"""
    if k <= 1 or not text:
        return [text] if text else []
    L = len(text)
    idx = [round(i * L / k) for i in range(k+1)]
    chunks = [text[idx[i]:idx[i+1]].strip() for i in range(k)]
    # 极短块合并（防空）
    chunks = [c for c in chunks if c]
    if not chunks:
        chunks = [text]
    return chunks

# -------- 词面工具 --------
def tokenize_basic(s: str) -> List[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", s.lower())

def jaccard(a: str, b: str) -> float:
    A = set(tokenize_basic(a))
    B = set(tokenize_basic(b))
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    inter = len(A & B); union = len(A | B)
    return inter / max(union, 1)

def _ngram(seq: List[str], n: int) -> List[Tuple[str,...]]:
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)] if len(seq) >= n else []

def rouge_n_f(ref_tok: List[str], hyp_tok: List[str], n: int=1) -> float:
    ref_ngrams = _ngram(ref_tok, n); hyp_ngrams = _ngram(hyp_tok, n)
    if not ref_ngrams or not hyp_ngrams: return 0.0
    from collections import defaultdict
    rc = defaultdict(int)
    for g in ref_ngrams: rc[g]+=1
    hc = defaultdict(int)
    for g in hyp_ngrams: hc[g]+=1
    overlap = sum(min(c, rc.get(g,0)) for g,c in hc.items())
    prec = overlap / max(len(hyp_ngrams), 1); rec = overlap / max(len(ref_ngrams), 1)
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

def lcs_len(a: List[str], b: List[str]) -> int:
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        ai=a[i-1]
        for j in range(1, len(b)+1):
            dp[i][j] = dp[i-1][j-1]+1 if ai==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def rouge_l_f(ref_tok: List[str], hyp_tok: List[str]) -> float:
    if not ref_tok or not hyp_tok: return 0.0
    L = lcs_len(ref_tok, hyp_tok)
    prec = L / max(len(hyp_tok),1); rec = L / max(len(ref_tok),1)
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

# -------- 相似度矩阵 & 软对齐 --------
def build_sim_matrix(S_units: List[str], T_units: List[str]) -> np.ndarray:
    if not S_units or not T_units:
        return np.zeros((len(S_units), len(T_units)), dtype=float)
    if USE_SK:
        vec = TfidfVectorizer(min_df=1, max_df=0.95)
        X = vec.fit_transform(S_units + T_units)
        Xs = X[:len(S_units)]
        Xt = X[len(S_units):]
        M = cosine_similarity(Xs, Xt)
        return np.clip(M, 0.0, 1.0)
    # fallback: Jaccard
    M = np.zeros((len(S_units), len(T_units)), dtype=float)
    S_tok = [set(tokenize_basic(s)) for s in S_units]
    T_tok = [set(tokenize_basic(t)) for t in T_units]
    for j, sj in enumerate(S_tok):
        for i, ti in enumerate(T_tok):
            if not sj and not ti:
                M[j,i]=0.0
            else:
                inter=len(sj & ti); union=len(sj | ti) if (sj or ti) else 1
                M[j,i]=inter/max(union,1)
    return M

def soft_alignment_A(M: np.ndarray, theta: float=0.05, tau: float=0.25) -> np.ndarray:
    if M.size==0:
        return np.zeros_like(M)
    mask = (M>theta).astype(float)
    logits = M / max(tau, 1e-8)
    logits = logits - (logits*mask).max(axis=1, keepdims=True)
    exps = np.exp(logits)*mask
    denom = exps.sum(axis=1, keepdims=True)+1e-12
    return exps/denom

def normalize_weights(w: List[float], L: int) -> np.ndarray:
    w = list(w)
    if L<=0: return np.zeros(0, dtype=float)
    if len(w)>=L: w=w[:L]
    else: w += [1e-8]*(L-len(w))
    arr = np.array(w, dtype=float)
    s = arr.sum()
    if s<=0: arr = np.ones(L, dtype=float); s=float(L)
    return arr/s

def weighted_coverage(A: np.ndarray, wT: np.ndarray, topq: float=0.2) -> Dict[str,float]:
    if A.size==0 or wT.size==0:
        return {"WCov":0.0,"WCov_TopQ":0.0,"Miss_TopQ":1.0}
    wT = wT/(wT.sum()+1e-12)
    covered = np.minimum(1.0, A.sum(axis=0))  # 每个 T 单元是否被覆盖
    WCov = float((wT*covered).sum())
    # Top-q（按权重质量）
    idx = np.argsort(-wT); cum=0.0; top=[]
    for i in idx:
        if cum>=topq-1e-12: break
        top.append(i); cum+=wT[i]
    if not top:
        WCov_TopQ=0.0
    else:
        top=np.array(top, dtype=int)
        WCov_TopQ=float((wT[top]*covered[top]).sum()/(wT[top].sum()+1e-12))
    return {"WCov":WCov, "WCov_TopQ":WCov_TopQ, "Miss_TopQ":1.0-WCov_TopQ}

# -------- 从消融日志聚合重要性 --------
def aggregate_importance(df_rows: List[Dict[str,Any]], granularity_target: str) -> Dict[Tuple[str,int], Dict[str,Any]]:
    """
    输入：同一 ablation 文件的记录列表；输出：(uuid, gen_index) -> {
        "imp": [importance per unit],  # 未归一
        "base_avg_logp": float,
        "K": int,  # 单元个数
    }
    逻辑：找到 base 的 avg_logp，然后对 drop_k 计算 max(0, base - drop_k)。
    """
    groups = defaultdict(list)
    for r in df_rows:
        if r.get("granularity") != granularity_target:
            continue
        u = r.get("uuid"); gi = int(r.get("gen_index",0))
        groups[(u,gi)].append(r)
    out = {}
    for key, items in groups.items():
        base = next((x for x in items if x.get("variant")=="base" and x.get("step_index",-1)==-1), None)
        if base is None:  # 没有基线，跳过
            continue
        base_lp = float(base.get("avg_logp", 0.0))
        # 收集 drop_k
        step_to_imp = {}
        for x in items:
            var = str(x.get("variant",""))
            if var.startswith("drop_"):
                k = int(x.get("step_index", -1))
                if k>=0:
                    drop_lp = float(x.get("avg_logp", base_lp))
                    imp = max(0.0, base_lp - drop_lp)  # 重要性定义
                    step_to_imp[k] = imp
        if not step_to_imp:
            continue
        # 规范化为 0..K-1 连续
        K = max(step_to_imp.keys())+1
        imp_vec = [step_to_imp.get(i, 0.0) for i in range(K)]
        out[key] = {"imp": imp_vec, "base_avg_logp": base_lp, "K": K}
    return out

# -------- 幻觉 / 外延（不需模型） --------
def hallucination_metrics(S_units: List[str], T_units: List[str], wS: Optional[np.ndarray], M: np.ndarray, theta_ext: float=0.20):
    m = len(S_units); n = len(T_units)
    if m==0:
        return {"Unmatched_Ratio":0.0,"Extraneous_Salience":0.0}
    if M.size==0:
        unmatched = np.ones(m, dtype=float)
    else:
        max_s = M.max(axis=1) if n>0 else np.zeros(m)
        unmatched = (max_s < theta_ext).astype(float)
    unmatched_ratio = float(unmatched.mean())
    if wS is None or len(wS)!=m:
        wS_ = np.ones(m, dtype=float)/m
    else:
        wS_ = wS/(wS.sum()+1e-12)
    extraneous = float((unmatched * wS_).sum())
    return {"Unmatched_Ratio": unmatched_ratio, "Extraneous_Salience": extraneous}

# -------- 主流程 --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_table", default="/root/autodl-fs/out2000/A+Q+RT/math/out_openr1_answer_summaries.jsonl")
    ap.add_argument("--summary_ablation", default="/root/autodl-fs/out2000/A+Q+RT/math/ablation_from_summary.jsonl")
    ap.add_argument("--reasoning_ablation", default="/root/autodl-fs/out200/ablation_results.jsonl")
    ap.add_argument("--out_csv", default="/root/autodl-fs/analysis/math_coverage_results.csv")
    ap.add_argument("--topq", type=float, default=0.2)
    ap.add_argument("--theta", type=float, default=0.05)   # 相似度阈
    ap.add_argument("--tau", type=float, default=0.25)     # 软对齐温度
    ap.add_argument("--ext_theta", type=float, default=0.20)  # 幻觉阈值
    args = ap.parse_args()

    # 1) 读表（只抽需要字段，省内存）
    sum_rows = []
    for r in read_jsonl(args.summary_table):
        if "uuid" in r and "summary_answer_driven" in r and "traces_natural" in r:
            sum_rows.append({
                "uuid": r["uuid"],
                "summary_text": r["summary_answer_driven"],
                "traces": r["traces_natural"],   # 列表[str]（按 gen_index 选择）
            })
    # 2) 消融→重要性（S: sentence；T: macro）
    sab_rows = list(read_jsonl(args.summary_ablation))
    res_rows = list(read_jsonl(args.reasoning_ablation))
    S_imp_map = aggregate_importance(sab_rows, granularity_target="sentence")
    T_imp_map = aggregate_importance(res_rows, granularity_target="macro")

    # 3) 构造交集：以 (uuid, gen_index) 为键
    #   注意：summary_table 只有 uuid 与 traces 列表，我们默认 gen_index=0（也可扩展：若 S_imp/T_imp 存在其他 gi，则优先该 gi）
    index_sum = {}
    for r in sum_rows:
        uuid = r["uuid"]
        index_sum[(uuid, 0)] = r   # 默认 gi=0；若你需要 gi!=0，按需扩展这里

    keys_inter = sorted(set(index_sum.keys()) & set(S_imp_map.keys()) & set(T_imp_map.keys()))
    rows = []
    for key in keys_inter:
        uuid, gi = key
        rec_sum = index_sum[key]
        s_text = rec_sum["summary_text"]
        t_list = rec_sum["traces"]
        if not isinstance(t_list, list) or len(t_list)<=gi:
            continue
        t_text = t_list[gi]

        # S 单元：分句
        S_units = split_sentences(s_text)
        if not S_units:
            continue
        # T 单元：按宏步数 K 等长切片
        K = T_imp_map[key]["K"]
        T_units = equal_chunks(t_text, K)
        if len(T_units) != K:
            # 若切片后段数与 K 不一致（极少），做微调：按 K 强制均分（包括空片段）
            T_units = equal_chunks(t_text + " ", K)  # 粗粘
            if not T_units:
                continue

        # 权重向量
        wS = normalize_weights(S_imp_map[key]["imp"], len(S_units)) if key in S_imp_map else None
        wT = normalize_weights(T_imp_map[key]["imp"], len(T_units))

        # 相似度与软对齐
        M = build_sim_matrix(S_units, T_units)
        A = soft_alignment_A(M, theta=args.theta, tau=args.tau)

        # 关键覆盖与错失
        cov = weighted_coverage(A, wT=wT, topq=args.topq)

        # 词面一致性（整段文本级）
        j = jaccard(s_text, t_text)
        ref = tokenize_basic(t_text); hyp = tokenize_basic(s_text)
        r1 = rouge_n_f(ref, hyp, 1); r2 = rouge_n_f(ref, hyp, 2); rl = rouge_l_f(ref, hyp)

        # 幻觉/外延（基于 S 句→T 单元最大相似阈）
        hall = hallucination_metrics(S_units, T_units, wS=wS, M=M, theta_ext=args.ext_theta)

        rows.append({
            "uuid": uuid,
            "gen_index": gi,
            "n_S_units": len(S_units),
            "n_T_units": len(T_units),
            "WCov": cov["WCov"],
            "WCov_TopQ": cov["WCov_TopQ"],
            "Miss_TopQ": cov["Miss_TopQ"],
            "Jaccard_ST": j,
            "ROUGE1_F": r1,
            "ROUGE2_F": r2,
            "ROUGEL_F": rl,
            "Unmatched_Ratio": hall["Unmatched_Ratio"],
            "Extraneous_Salience": hall["Extraneous_Salience"],
        })

    df = pd.DataFrame(rows).sort_values("Miss_TopQ", ascending=False)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] 样本数={len(df)}  已保存：{args.out_csv}")
    if len(df):
        print(df.head(10))
        print(df[["WCov","WCov_TopQ","Miss_TopQ","Jaccard_ST","ROUGE1_F","ROUGE2_F","ROUGEL_F",
                  "Unmatched_Ratio","Extraneous_Salience"]].describe())

if __name__ == "__main__":
    main()
