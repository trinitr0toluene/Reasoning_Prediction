import os
import json
import re
import argparse
import random
import hashlib
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# ---------------- IO ----------------

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ----------- Join-key helpers (robust) -----------

def _to_int_or_none(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None

def best_join_keys(d: Dict[str, Any]) -> List[Tuple]:
    """
    Return candidate join keys from strongest to weakest.
    We DO NOT assume gen_index is present nor numeric.
    """
    keys = []
    uid = d.get("uuid")
    sid = d.get("sample_id")
    gi  = _to_int_or_none(d.get("gen_index"))
    iid = d.get("id")
    qid = d.get("qid")

    if uid is not None and gi is not None:
        keys.append(("uuid+gen", str(uid), gi))
    if sid is not None and gi is not None:
        keys.append(("sid+gen", str(sid), gi))

    if uid is not None:
        keys.append(("uuid", str(uid)))
    if sid is not None:
        keys.append(("sid", str(sid)))
    if iid is not None:
        keys.append(("id", str(iid)))
    if qid is not None:
        keys.append(("qid", str(qid)))

    q = d.get("prompt") or d.get("question") or ""
    keys.append(("hash", hashlib.md5(q.encode("utf-8")).hexdigest()))
    return keys

def make_key(d: Dict[str, Any]):
    return best_join_keys(d)[0]

def group_key_of(d: Dict[str, Any]) -> str:
    """
    Grouping key for train/valid split: prefer stable per-question ids.
    DO NOT use gen_index here to avoid splitting variants of the same question.
    """
    for name in ("uuid", "sample_id", "id", "qid"):
        v = d.get(name)
        if v is not None and v != "":
            return f"{name}:{str(v)}"
    q = d.get("prompt") or d.get("question") or ""
    return "hash:" + hashlib.md5(q.encode("utf-8")).hexdigest()

def extract_summary_text(d: Dict[str, Any]) -> str:
    for k in ("summary", "summary_text", "summary_answer_driven"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def build_summary_index(path: str):
    """
    Build summary index allowing multiple candidate keys to point to the same summary.
    This makes alignment robust when gen_index is missing in either side.
    """
    idx = {}
    if not path:
        return idx
    for row in read_jsonl(path):
        s = extract_summary_text(row)
        if not s:
            continue
        for k in best_join_keys(row):
            if k not in idx:
                idx[k] = s
    return idx

# ------------- split -------------

def split_records(records: List[Dict[str, Any]], val_ratio=0.1, kfold=0, fold_index=0, seed=42):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[group_key_of(r)].append(r)
    keys = list(groups.keys())
    rnd = random.Random(seed); rnd.shuffle(keys)
    if kfold and kfold > 1:
        fold_size = max(1, len(keys)//kfold)
        val_keys = set(keys[fold_index*fold_size : (fold_index+1)*fold_size])
    else:
        cut = max(1, int(len(keys)*val_ratio))
        val_keys = set(keys[:cut])
    train, valid = [], []
    for k in keys:
        (valid if k in val_keys else train).extend(groups[k])
    return train, valid

# ------------- NEW: subsampling -------------

def sample_records(records: List[Dict[str, Any]],
                   max_samples: int = 0,
                   sample_ratio: float = 1.0,
                   grouped: bool = False,
                   seed: int = 42) -> List[Dict[str, Any]]:
    """Return a (possibly) downsampled copy of records.
       - If grouped=True, sample by group_key_of to avoid mixing variants of same question.
       - If both max_samples and sample_ratio provided, we take the stricter target size.
    """
    n = len(records)
    if n == 0:
        return records

    target_by_ratio = int(n * sample_ratio) if (0 < sample_ratio < 1.0) else n
    target = target_by_ratio
    if max_samples and max_samples > 0:
        target = min(target_by_ratio, max_samples)
    target = max(1, min(n, target))
    if target >= n:
        return records

    rnd = random.Random(seed)

    if not grouped:
        return rnd.sample(records, target)

    # grouped sampling: take whole groups to avoid leakage
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[group_key_of(r)].append(r)

    keys = list(groups.keys())
    rnd.shuffle(keys)

    out, tot = [], 0
    for k in keys:
        g = groups[k]
        if tot + len(g) > target and tot > 0:
            continue
        out.extend(g)
        tot += len(g)
        if tot >= target:
            break

    if tot < target:
        # optional top-up: take a few extra samples from remaining groups to hit target
        remain = []
        selected = set(id(x) for x in out)
        for k in keys:
            for rec in groups[k]:
                if id(rec) not in selected:
                    remain.append(rec)
        need = target - tot
        if need > 0 and remain:
            out.extend(rnd.sample(remain, min(need, len(remain))))

    return out[:target]

# ---------- <think> parsing ----------

_THINK_TAG_RE = re.compile(r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)
_END_THINK_RE  = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)

def extract_think_and_rest(text: str):
    """Return (list_of_thinks, visible_text_after_last_think)."""
    if not isinstance(text, str) or not text:
        return [], ""
    thinks = _THINK_TAG_RE.findall(text)
    last_end = 0
    for m in _END_THINK_RE.finditer(text):
        last_end = m.end()
    rest = text[last_end:].strip() if last_end else text.strip()
    return thinks, rest

def strip_think_keep_visible(s: str) -> str:
    _, rest = extract_think_and_rest(s or "")
    return rest or (s or "").strip()

# ---------- pack-three fallback ----------

def _pack_three_fallback(tokenizer, segA: str, segB: str, segC: str, max_len: int, ratio=(4,4,2), headroom=8):
    idsA = tokenizer.encode(segA, add_special_tokens=False)
    idsB = tokenizer.encode(segB, add_special_tokens=False)
    idsC = tokenizer.encode(segC, add_special_tokens=False)
    bA, bB, bC = ratio; cap = max(16, max_len - headroom)
    qA = max(8, cap * bA // (bA+bB+bC)); qB = max(8, cap * bB // (bA+bB+bC)); qC = max(8, cap * bC // (bA+bB+bC))
    def cut(arr, q): return arr[:max(0, min(q, len(arr)))]
    A, B, C = cut(idsA,qA), cut(idsB,qB), cut(idsC,qC)
    used = len(A)+len(B)+len(C); left = cap-used
    if left>0:
        rema = [("A", idsA[len(A):]), ("B", idsB[len(B):]), ("C", idsC[len(C):])]
        while left>0:
            rema.sort(key=lambda x: len(x[1]), reverse=True)
            name, rem = rema[0]
            if not rem: break
            take = min(left, len(rem))
            if name=="A": A += rem[:take]
            elif name=="B": B += rem[:take]
            else: C += rem[:take]
            rema[0] = (name, rem[take:])
            left -= take
    return (
        tokenizer.decode(A, skip_special_tokens=True),
        tokenizer.decode(B, skip_special_tokens=True),
        tokenizer.decode(C, skip_special_tokens=True),
    )

# ------------- dataset -------------

def _safe_float(y):
    try:
        return float(str(y).strip())
    except Exception:
        return None

class RLPDataset(Dataset):
    def __init__(self, path: str=None, tokenizer=None, max_length: int=1024,
                 use_summary: bool=False, summary_index: Dict[Any, str]=None,
                 records: List[Dict[str, Any]] = None):
        self.tok = tokenizer
        self.max_length = max_length
        self.use_summary = use_summary
        self.sidx = summary_index or {}
        if records is not None:
            self.records = records
        else:
            self.records = list(read_jsonl(path))

    def __len__(self):
        return len(self.records)

    def _pick_answer(self, r: Dict[str, Any]) -> str:
        if "final_answer" in r and isinstance(r["final_answer"], str):
            return strip_think_keep_visible(r["final_answer"])
        if "answer" in r and isinstance(r["answer"], str):
            return strip_think_keep_visible(r["answer"])
        out = r.get("output", "")
        return strip_think_keep_visible(out if isinstance(out, str) else str(out))

    def _lookup_summary(self, r: Dict[str, Any]) -> str:
        # Use record-bundled summary if present
        for k in ("summary", "summary_text", "summary_answer_driven"):
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        if not self.sidx:
            return ""
        for k in best_join_keys(r):
            if k in self.sidx:
                return self.sidx[k]
        return ""

    def _pack_three(self, A, B, C):
        try:
            return pack_three_segments(self.tok, A, B, C, self.max_length)  # if user provided
        except NameError:
            return _pack_three_fallback(self.tok, A, B, C, self.max_length)

    def _build_text(self, r: Dict[str, Any]) -> str:
        prompt = r.get("prompt", r.get("question", "")) or ""
        answer = self._pick_answer(r)
        s_txt = self._lookup_summary(r) if self.use_summary else ""
        if self.use_summary and s_txt:
            A = f"Prompt:\n{prompt}"
            B = f"Answer:\n{answer}"
            C = f"Summary:\n{s_txt}"
            A, B, C = self._pack_three(A, B, C)
            return f"{A}\n\n{B}\n\n{C}"
        return f"Prompt:\n{prompt}\n\nAnswer:\n{answer}"

    def __getitem__(self, i):
        r = self.records[i]
        x = self._build_text(r)

        y = r.get("reasoning_token_count", None)
        if y is None:
            for k in ("reasoning_tokens", "reasoning_len", "hidden_tokens", "hidden_len"):
                if k in r and r[k] is not None:
                    y = r[k]; break
        y = _safe_float(y)
        if y is None:
            raise ValueError("Missing target: reasoning_token_count")

        enc = self.tok(x, truncation=True, max_length=self.max_length, padding=False, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["y"] = float(y)
        return item

# -------------- model --------------

class MeanPooler(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        s = (last_hidden_state * mask).sum(dim=1)
        d = mask.sum(dim=1).clamp(min=1e-6)
        return s / d

class RLPRegressor(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1, **fp_kwargs):
        super().__init__()
        self.model_name = model_name
        # 优先 ForCausalLM，之后再解包底座
        try:
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, **fp_kwargs
            )
        except Exception:
            self.backbone = AutoModel.from_pretrained(
                model_name, trust_remote_code=True, **fp_kwargs
            )
        hidden = self._infer_hidden_size(self.backbone)
        self.pool = MeanPooler()
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1)
        )

    @staticmethod
    def _infer_hidden_size(bb):
        # 尽量通过 config 拿 hidden_size
        cfg = getattr(bb, "config", None)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            return cfg.hidden_size
        # 少数模型名字不同
        for k in ("n_embd", "d_model", "hidden_sizes"):
            if hasattr(cfg, k):
                v = getattr(cfg, k)
                return v[-1] if isinstance(v, (list, tuple)) else int(v)
        raise RuntimeError("Cannot infer hidden size from backbone config.")

    def _unwrap_base(self):
        """
        返回真正的“底座”模型（不含 LM 头），兼容：
        - 原生 ForCausalLM: .model / .transformer / .base_model
        - PEFT: .get_base_model()
        - 纯 AutoModel: 直接返回自身
        """
        bb = self.backbone
        # 先解 PEFT
        if hasattr(bb, "get_base_model"):
            try:
                bb = bb.get_base_model()
            except Exception:
                pass
        # 常见字段：Qwen/Llama/Mistral 等
        for name in ("model", "transformer", "base_model", "backbone"):
            if hasattr(bb, name) and getattr(bb, name) is not None:
                return getattr(bb, name)
        return bb  # 已经是 AutoModel 的情况

    def forward(self, input_ids, attention_mask):
        base = self._unwrap_base()

        # 1) 尝试直接拿 last_hidden_state（显存友好：不打开 hidden_states）
        out = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = getattr(out, "last_hidden_state", None)

        # 2) 若拿不到，则退回打开 hidden_states 获取最后一层
        if last_hidden is None:
            out = base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            if getattr(out, "last_hidden_state", None) is not None:
                last_hidden = out.last_hidden_state
            elif getattr(out, "hidden_states", None) is not None:
                last_hidden = out.hidden_states[-1]
            else:
                raise RuntimeError("Backbone produced neither last_hidden_state nor hidden_states.")

        z = self.pool(last_hidden, attention_mask)   # [B, H]
        return self.head(z).squeeze(-1)              # [B]

# -------------- losses & eval --------------

def huber_relative(pred, target, delta=0.1, eps=1e-6):
    rel = (pred - target).abs() / (target.abs().clamp(min=eps))
    quad = torch.minimum(rel, torch.tensor(delta, device=rel.device))
    lin = rel - quad
    return (0.5*quad**2 + delta*lin).mean()

def aggregated_bias_regularizer(pred, target, eps=1e-6):
    num = (pred - target).sum()
    den = target.abs().sum().clamp(min=eps)
    return (num.abs() / den)

@torch.no_grad()
def evaluate(model, loader, device, rel_thr=0.33):
    model.eval(); ys, yh = [], []
    for b in loader:
        p = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        ys.append(b["y"]); yh.append(p.cpu())
    y = torch.cat(ys).float(); p = torch.cat(yh).float()
    rel = (p - y).abs() / y.abs().clamp(min=1e-6)
    return {
        "pass@1": (rel<=rel_thr).float().mean().item(),
        "avg_error": (p - y).abs().mean().item(),
        "aggregated_error": (p - y).sum().abs().div(y.abs().sum().clamp(min=1e-6)).item(),
    }

# -------------- utils --------------

def set_seed(seed:int):
    random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -------------- main --------------

def parse_args():
    ap = argparse.ArgumentParser()

    # Single-file or traditional split
    ap.add_argument("--data-file", type=str, default="/root/autodl-fs/openr1_math_all.jsonl", help="Single JSONL; split in code")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--kfold", type=int, default=0)
    ap.add_argument("--fold-index", type=int, default=0)
    ap.add_argument("--train-file", type=str, default="")
    ap.add_argument("--valid-file", type=str, default="")

    ap.add_argument("--model-name", type=str, default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct")
    ap.add_argument("--summary-file", type=str, default="/root/autodl-fs/out2000/A+Q+RT/math/out_openr1_answer_summaries.jsonl")
    ap.add_argument("--use-summary", action="store_true")
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lambda-aggr", type=float, default=0.2)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="/root/autodl-fs/runs/rlp_regressor")
    ap.add_argument("--tokenizer", type=str, default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct", help="Tokenizer repo/id 或本地路径")

    # LoRA
    ap.add_argument("--use-lora", action="store_true", help="enable LoRA instead of full-finetune")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-target",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="comma-separated module names to inject LoRA into",
    )
    ap.add_argument("--save-lora-only", action="store_true", help="only save LoRA adapter (and our head)")

    # NEW: subsampling
    ap.add_argument("--max-samples", type=int, default=2000, help="cap total samples; 0 = use all")
    ap.add_argument("--sample-ratio", type=float, default=1.0, help="fraction of data to use (0,1]; ignored if >=1")
    ap.add_argument("--grouped-sampling", action="store_true", help="sample by question groups to avoid leakage")
    ap.add_argument("--sample-seed", type=int, default=42, help="rng seed for subsampling")

    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tok_name = args.tokenizer if args.tokenizer else args.model_name
    local_only = os.path.isdir(tok_name)
    tok = AutoTokenizer.from_pretrained(
        tok_name,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=local_only
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    sidx = build_summary_index(args.summary_file)

    # Build datasets
    if args.data_file:
        all_recs = list(read_jsonl(args.data_file))
        # === Subsample before split (保持验证代表性) ===
        all_recs = sample_records(
            all_recs,
            max_samples=args.max_samples,
            sample_ratio=args.sample_ratio,
            grouped=args.grouped_sampling,
            seed=args.sample_seed
        )
        print(f"[info] sampled {len(all_recs)} records from original pool")
        train_recs, valid_recs = split_records(all_recs, args.val_ratio, args.kfold, args.fold_index, args.seed)
        train_ds = RLPDataset(None, tok, args.max_length, args.use_summary, sidx, train_recs)
        valid_ds = RLPDataset(None, tok, args.max_length, args.use_summary, sidx, valid_recs)
    else:
        if not (args.train_file and args.valid_file):
            raise ValueError("Either --data-file or --train-file/--valid-file must be provided.")
        train_recs = list(read_jsonl(args.train_file))
        valid_recs = list(read_jsonl(args.valid_file))
        # 只对训练集做子采样
        train_recs = sample_records(
            train_recs,
            max_samples=args.max_samples,
            sample_ratio=args.sample_ratio,
            grouped=args.grouped_sampling,
            seed=args.sample_seed
        )
        print(f"[info] sampled train -> {len(train_recs)} | valid -> {len(valid_recs)}")
        train_ds = RLPDataset(None, tok, args.max_length, args.use_summary, sidx, train_recs)
        valid_ds = RLPDataset(None, tok, args.max_length, args.use_summary, sidx, valid_recs)

    base_collate = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    def collate_with_y(features):
        ys = torch.tensor([_safe_float(f["y"]) for f in features], dtype=torch.float)
        for f in features:
            f.pop("y", None)
        batch = base_collate(features)
        batch["y"] = ys
        return batch

    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_with_y)
    va = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_with_y)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RLPRegressor(args.model_name).to(dev)

    if hasattr(model.backbone, "config") and hasattr(model.backbone.config, "use_cache"):
        model.backbone.config.use_cache = False

    # # 可选：开启梯度检查点（以算换存）
    # if args.grad_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
    #     model.backbone.gradient_checkpointing_enable()

    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # k-bit 量化预处理（若已量化加载）
        quantized = any([
            hasattr(model.backbone, "is_loaded_in_8bit") and model.backbone.is_loaded_in_8bit,
            hasattr(model.backbone, "is_loaded_in_4bit") and model.backbone.is_loaded_in_4bit,
        ])
        if quantized:
            model.backbone = prepare_model_for_kbit_training(model.backbone)

        # 自动推断目标模块
        all_names = [n for n, _ in model.backbone.named_modules()]
        def any_in(names, keys):
            return any(any(k in n for n in names) for k in keys)

        is_clm = hasattr(model.backbone.config, "is_decoder") and model.backbone.config.is_decoder
        task_type = "CAUSAL_LM" if is_clm else "FEATURE_EXTRACTION"

        if any_in(all_names, ["q_proj","k_proj","v_proj","o_proj"]):
            targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
            task_type = "CAUSAL_LM"
        elif any_in(all_names, ["W_pack","query_key_value"]):
            cand = []
            if any_in(all_names, ["W_pack"]): cand += ["W_pack","o_proj"]
            if any_in(all_names, ["query_key_value"]): cand += ["query_key_value","dense"]
            targets = list(dict.fromkeys(cand + ["gate_proj","up_proj","down_proj"]))
            task_type = "CAUSAL_LM"
        elif any_in(all_names, ["query","key","value","dense"]):
            targets = ["query","key","value","dense"]
            task_type = "FEATURE_EXTRACTION"
        else:
            targets = [t.strip() for t in args.lora_target.split(",") if t.strip()]

        lconf = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=targets,
        )
        model.backbone = get_peft_model(model.backbone, lconf)

        # 冻结除 LoRA 以外的骨干权重，仅训 LoRA + 回归头
        for n, p in model.named_parameters():
            if n.startswith("backbone.") and "lora_" not in n:
                p.requires_grad_(False)

        try:
            model.backbone.print_trainable_parameters()
        except Exception:
            pass
    # ===== end LoRA wrap =====

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(grouped, lr=args.lr)
    steps = max(1, len(tr) * args.epochs); warm = int(steps * args.warmup_ratio)
    sch = get_linear_schedule_with_warmup(opt, warm, steps)

    best = None
    for ep in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(tr, desc=f"epoch {ep}")
        for b in pbar:
            opt.zero_grad(set_to_none=True)
            pred = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
            y = b["y"].to(dev)
            loss_main = huber_relative(pred, y)
            loss_aggr = aggregated_bias_regularizer(pred, y)
            loss = loss_main + args.lambda_aggr * loss_aggr
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "main": f"{loss_main.item():.4f}", "aggr": f"{loss_aggr.item():.4f}"})

        m = evaluate(model, va, dev)
        print(f"[valid] pass@1={m['pass@1']:.4f} avg={m['avg_error']:.2f} aggr={m['aggregated_error']:.4f}")
        score = m["aggregated_error"]
        if best is None or score < best:
            best = score
            torch.save(
                {"model": model.state_dict(), "tokenizer": args.model_name, "args": vars(args), "metrics": m},
                os.path.join(args.output_dir, "best.pt")
            )
            print("Saved best.")
            if args.use_lora and args.save_lora_only:
                lora_dir = os.path.join(args.output_dir, "lora_adapter")
                os.makedirs(lora_dir, exist_ok=True)
                try:
                    model.backbone.save_pretrained(lora_dir)
                    print(f"[info] Saved LoRA adapter to {lora_dir}")
                except Exception as e:
                    print("[warn] failed to save LoRA adapter:", e)

    torch.save({"model": model.state_dict(), "tokenizer": args.model_name, "args": vars(args)},
               os.path.join(args.output_dir, "last.pt"))

if __name__ == "__main__":
    main()