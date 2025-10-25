# -*- coding: utf-8 -*-
"""
ft_classify.py
--------------
Classification baseline for reasoning-length prediction (bucketed targets).
- Same data/summary/subsampling pipeline as fine_tune.py
- Same backbone + mean-pooling encoder, but with a classification head
- Saves class bin metadata so eval.py can convert class → numeric center

Usage (example):
python ft_classify.py \
  --data-file /root/autodl-fs/openr1_math_all.jsonl \
  --val-ratio 0.1 \
  --model-name /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
  --tokenizer  /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
  --use-lora \
  --max-samples 20000 --grouped-sampling --sample-seed 42 \
  --batch-size 1 --max-length 512 \
  --epochs 2 \
  --bin-edges 16,32,64,128,256,512,1024,2048,4096,8192,16384 \
  --output-dir runs/cls_20k_seed42
"""
import os, json, re, argparse, random, hashlib
from typing import Any, Dict, List, Tuple
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    DataCollatorWithPadding, get_linear_schedule_with_warmup
)
from tqdm import tqdm

# ---------------- IO ----------------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

# ----------- Join-key helpers (robust) -----------
def _to_int_or_none(x):
    try: return int(x)
    except (TypeError, ValueError): return None

def best_join_keys(d: Dict[str, Any]) -> List[Tuple]:
    keys = []
    uid = d.get("uuid"); sid = d.get("sample_id"); gi = _to_int_or_none(d.get("gen_index"))
    iid = d.get("id"); qid = d.get("qid")
    if uid is not None and gi is not None: keys.append(("uuid+gen", str(uid), gi))
    if sid is not None and gi is not None: keys.append(("sid+gen", str(sid), gi))
    if uid is not None: keys.append(("uuid", str(uid)))
    if sid is not None: keys.append(("sid", str(sid)))
    if iid is not None: keys.append(("id", str(iid)))
    if qid is not None: keys.append(("qid", str(qid)))
    q = d.get("prompt") or d.get("question") or ""
    keys.append(("hash", hashlib.md5(q.encode("utf-8")).hexdigest()))
    return keys

def group_key_of(d: Dict[str, Any]) -> str:
    for name in ("uuid","sample_id","id","qid"):
        v = d.get(name)
        if v not in (None, ""): return f"{name}:{str(v)}"
    q = d.get("prompt") or d.get("question") or ""
    return "hash:" + hashlib.md5(q.encode("utf-8")).hexdigest()

def extract_summary_text(d: Dict[str, Any]) -> str:
    for k in ("summary","summary_text","summary_answer_driven"):
        v = d.get(k)
        if isinstance(v, str) and v.strip(): return v.strip()
    return ""

def build_summary_index(path: str):
    idx = {}
    if not path: return idx
    for row in read_jsonl(path):
        s = extract_summary_text(row)
        if not s: continue
        for k in best_join_keys(row):
            if k not in idx: idx[k] = s
    return idx

# ------------- split & sampling -------------
def split_records(records, val_ratio=0.1, kfold=0, fold_index=0, seed=42):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records: groups[group_key_of(r)].append(r)
    keys = list(groups.keys()); rnd = random.Random(seed); rnd.shuffle(keys)
    if kfold and kfold>1:
        fold_size = max(1, len(keys)//kfold); val_keys = set(keys[fold_index*fold_size:(fold_index+1)*fold_size])
    else:
        cut = max(1, int(len(keys)*val_ratio)); val_keys = set(keys[:cut])
    train, valid = [], []
    for k in keys: (valid if k in val_keys else train).extend(groups[k])
    return train, valid

def sample_records(records, max_samples=0, sample_ratio=1.0, grouped=False, seed=42):
    n = len(records)
    if n==0: return records
    target_by_ratio = int(n*sample_ratio) if (0<sample_ratio<1.0) else n
    target = min(target_by_ratio, max_samples) if max_samples>0 else target_by_ratio
    target = max(1, min(n, target))
    if target>=n: return records
    rnd = random.Random(seed)
    if not grouped: return rnd.sample(records, target)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records: groups[group_key_of(r)].append(r)
    keys = list(groups); rnd.shuffle(keys)
    out, tot = [], 0
    for k in keys:
        g = groups[k]
        if tot+len(g)>target and tot>0: continue
        out.extend(g); tot+=len(g)
        if tot>=target: break
    if tot<target:
        remain = []
        sel = set(id(x) for x in out)
        for k in keys:
            for r in groups[k]:
                if id(r) not in sel: remain.append(r)
        need = target-tot
        if need>0 and remain: out.extend(rnd.sample(remain, min(need, len(remain))))
    return out[:target]

# ---------- <think> parsing ----------
_THINK_TAG_RE = re.compile(r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.IGNORECASE|re.DOTALL)
_END_THINK_RE  = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
def extract_think_and_rest(text: str):
    if not isinstance(text, str) or not text: return [], ""
    thinks = _THINK_TAG_RE.findall(text)
    last_end = 0
    for m in _END_THINK_RE.finditer(text): last_end = m.end()
    rest = text[last_end:].strip() if last_end else text.strip()
    return thinks, rest
def strip_think_keep_visible(s: str) -> str:
    _, rest = extract_think_and_rest(s or ""); return rest or (s or "").strip()

# ---------- pack-three fallback ----------
def _pack_three_fallback(tokenizer, A, B, C, max_len, ratio=(4,4,2), headroom=8):
    idsA = tokenizer.encode(A, add_special_tokens=False)
    idsB = tokenizer.encode(B, add_special_tokens=False)
    idsC = tokenizer.encode(C, add_special_tokens=False)
    bA,bB,bC = ratio; cap = max(16, max_len - headroom)
    qA = max(8, cap*bA//(bA+bB+bC)); qB = max(8, cap*bB//(bA+bB+bC)); qC = max(8, cap*bC//(bA+bB+bC))
    def cut(x,q): return x[:max(0, min(q, len(x)))]
    A,B,C = cut(idsA,qA), cut(idsB,qB), cut(idsC,qC)
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
            rema[0] = (name, rem[take:]); left -= take
    tok = tokenizer
    return tok.decode(A, skip_special_tokens=True), tok.decode(B, skip_special_tokens=True), tok.decode(C, skip_special_tokens=True)

# ------------- dataset -------------
def _safe_float(y):
    try: return float(str(y).strip())
    except Exception: return None

class RLPDataset(Dataset):
    def __init__(self, path=None, tokenizer=None, max_length=1024,
                 use_summary=False, summary_index=None, records=None,
                 bin_edges=None):
        self.tok = tokenizer
        self.max_length = max_length
        self.use_summary = use_summary
        self.sidx = summary_index or {}
        self.bin_edges = bin_edges  # ascending list
        if records is not None: self.records = records
        else: self.records = list(read_jsonl(path))

    def __len__(self): return len(self.records)

    def _pick_answer(self, r):
        if "final_answer" in r and isinstance(r["final_answer"], str):
            return strip_think_keep_visible(r["final_answer"])
        if "answer" in r and isinstance(r["answer"], str):
            return strip_think_keep_visible(r["answer"])
        out = r.get("output", "")
        return strip_think_keep_visible(out if isinstance(out, str) else str(out))

    def _lookup_summary(self, r):
        for k in ("summary","summary_text","summary_answer_driven"):
            v = r.get(k)
            if isinstance(v, str) and v.strip(): return v.strip()
        if not self.sidx: return ""
        for k in best_join_keys(r):
            if k in self.sidx: return self.sidx[k]
        return ""

    def _pack_three(self, A,B,C):
        try:
            return pack_three_segments(self.tok, A,B,C, self.max_length)
        except NameError:
            return _pack_three_fallback(self.tok, A,B,C, self.max_length)

    def _build_text(self, r):
        prompt = r.get("prompt", r.get("question", "")) or ""
        answer = self._pick_answer(r)
        s_txt = self._lookup_summary(r) if self.use_summary else ""
        if self.use_summary and s_txt:
            A=f"Prompt:\n{prompt}"; B=f"Answer:\n{answer}"; C=f"Summary:\n{s_txt}"
            A,B,C = self._pack_three(A,B,C); return f"{A}\n\n{B}\n\n{C}"
        return f"Prompt:\n{prompt}\n\nAnswer:\n{answer}"

    def _to_class(self, y):
        # y is float token count; map to bin index using self.bin_edges (<= edge)
        edges = self.bin_edges
        if edges is None or len(edges)==0:
            raise ValueError("bin_edges required in RLPDataset for classification.")
        # bins: ( -inf, e1 ], (e1, e2 ], ...  (right-closed)
        for i, e in enumerate(edges):
            if y <= e: return i
        return len(edges) - 1  # clamp to last

    def __getitem__(self, i):
        r = self.records[i]
        x = self._build_text(r)

        y = r.get("reasoning_token_count", None)
        if y is None:
            for k in ("reasoning_tokens", "reasoning_len", "hidden_tokens", "hidden_len"):
                if k in r and r[k] is not None:
                    y = r[k]; break
        y = _safe_float(y)
        if y is None: raise ValueError("Missing target: reasoning_token_count")

        enc = self.tok(x, truncation=True, max_length=self.max_length, padding=False, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["y_cont"] = float(y)
        item["y_cls"] = int(self._to_class(float(y)))
        return item

# -------------- model --------------
class MeanPooler(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        s = (last_hidden_state * mask).sum(dim=1)
        d = mask.sum(dim=1).clamp(min=1e-6)
        return s / d

class RLPClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1, **fp_kwargs):
        super().__init__()
        self.model_name = model_name
        # Prefer ForCausalLM then unwrap base later
        try:
            self.backbone = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **fp_kwargs)
        except Exception:
            self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True, **fp_kwargs)
        hidden = self._infer_hidden_size(self.backbone)
        self.pool = MeanPooler()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, num_classes),
        )

    @staticmethod
    def _infer_hidden_size(bb):
        cfg = getattr(bb, "config", None)
        for k in ("hidden_size","n_embd","d_model"):
            if cfg is not None and hasattr(cfg, k):
                v = getattr(cfg, k)
                return v[-1] if isinstance(v, (list,tuple)) else int(v)
        raise RuntimeError("Cannot infer hidden size from backbone config.")

    def _unwrap_base(self):
        bb = self.backbone
        if hasattr(bb, "get_base_model"):
            try: bb = bb.get_base_model()
            except Exception: pass
        for name in ("model","transformer","base_model","backbone"):
            if hasattr(bb, name) and getattr(bb, name) is not None:
                return getattr(bb, name)
        return bb

    def forward(self, input_ids, attention_mask):
        base = self._unwrap_base()
        out = base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = getattr(out, "last_hidden_state", None)
        if last_hidden is None:
            out = base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            last_hidden = out.last_hidden_state if getattr(out, "last_hidden_state", None) is not None else out.hidden_states[-1]
        z = self.pool(last_hidden, attention_mask)
        return self.head(z)  # [B, num_classes]

# -------------- metrics --------------
@torch.no_grad()
def evaluate_cls(model, loader, device, class_centers: List[float], rel_thr=0.33):
    model.eval(); ys, yh_ids = [], []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        pred = logits.argmax(dim=-1).cpu()
        yh_ids.append(pred); ys.append(b["y_cont"])
    y = torch.cat(ys).float()
    yhat_ids = torch.cat(yh_ids).long()
    centers = torch.tensor(class_centers, dtype=torch.float)
    p = centers[yhat_ids]  # [N] numeric prediction via bin centers
    rel = (p - y).abs() / y.abs().clamp(min=1e-6)
    cls_acc = (yhat_ids == torch.tensor([int(c) for c in torch.cat([b['y_cls'] for b in loader.dataset[:1]])], dtype=torch.long)).float().mean().item() if False else None
    return {
        "pass@1": (rel<=rel_thr).float().mean().item(),
        "avg_error": (p - y).abs().mean().item(),
        "aggregated_error": (p - y).sum().abs().div(y.abs().sum().clamp(min=1e-6)).item(),
        # 分类准确率可在 eval.py 内另算（这里先不返回）
    }

# -------------- utils --------------
def set_seed(seed:int):
    random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -------------- main --------------
def parse_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data-file", type=str, default="/root/autodl-fs/openr1_math_all.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--kfold", type=int, default=0)
    ap.add_argument("--fold-index", type=int, default=0)
    ap.add_argument("--train-file", type=str, default="")
    ap.add_argument("--valid-file", type=str, default="")
    # model
    ap.add_argument("--model-name", type=str, default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct")
    ap.add_argument("--tokenizer", type=str, default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct")
    ap.add_argument("--use-summary", action="store_true")
    ap.add_argument("--summary-file", type=str, default="/root/autodl-fs/out2000/A+Q+RT/math/out_openr1_answer_summaries.jsonl")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="runs/rlp_classifier")
    # LoRA
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--save-lora-only", action="store_true")
    # subsampling
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--sample-ratio", type=float, default=1.0)
    ap.add_argument("--grouped-sampling", action="store_true")
    ap.add_argument("--sample-seed", type=int, default=42)
    # binning
    ap.add_argument("--bin-edges", type=str, default="16,32,64,128,256,512,1024,2048,4096,8192,16384",
                    help="ascending edges (right-closed). N edges => N classes.")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tok_name = args.tokenizer or args.model_name
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True, trust_remote_code=True,
                                        local_files_only=os.path.isdir(tok_name))
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    sidx = build_summary_index(args.summary_file)

    # load & sample
    if args.data_file:
        all_recs = list(read_jsonl(args.data_file))
        all_recs = sample_records(all_recs, args.max_samples, args.sample_ratio, args.grouped_sampling, args.sample_seed)
        print(f"[info] sampled {len(all_recs)} records from original pool")
        train_recs, valid_recs = split_records(all_recs, args.val_ratio, args.kfold, args.fold_index, args.seed)
    else:
        if not (args.train_file and args.valid_file):
            raise ValueError("Either --data-file or --train-file/--valid-file must be provided.")
        train_recs = list(read_jsonl(args.train_file))
        valid_recs = list(read_jsonl(args.valid_file))
        train_recs = sample_records(train_recs, args.max_samples, args.sample_ratio, args.grouped_sampling, args.sample_seed)

    # bins
    bin_edges = [float(x) for x in args.bin-edges.split(",")] if hasattr(args, "bin-edges") else [float(x) for x in args.bin_edges.split(",")]
    class_centers = []
    low = 0.0
    for e in bin_edges:
        high = e
        class_centers.append( (low + high) / 2.0 )
        low = high
    num_classes = len(bin_edges)
    print(f"[bins] edges={bin_edges} | num_classes={num_classes}")

    # datasets
    train_ds = RLPDataset(records=train_recs, tokenizer=tok, max_length=args.max_length,
                          use_summary=args.use_summary, summary_index=sidx, bin_edges=bin_edges)
    valid_ds = RLPDataset(records=valid_recs, tokenizer=tok, max_length=args.max_length,
                          use_summary=args.use_summary, summary_index=sidx, bin_edges=bin_edges)

    base_collate = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
    def collate_with_y(features):
        y_cont = torch.tensor([float(f.pop("y_cont")) for f in features], dtype=torch.float)
        y_cls  = torch.tensor([int(f.pop("y_cls"))  for f in features], dtype=torch.long)
        batch = base_collate(features); batch["y_cont"] = y_cont; batch["y_cls"] = y_cls
        return batch

    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_with_y)
    va = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_with_y)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RLPClassifier(args.model_name, num_classes=num_classes).to(dev)

    # LoRA
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        quantized = any([
            hasattr(model.backbone, "is_loaded_in_8bit") and model.backbone.is_loaded_in_8bit,
            hasattr(model.backbone, "is_loaded_in_4bit") and model.backbone.is_loaded_in_4bit,
        ])
        if quantized:
            model.backbone = prepare_model_for_kbit_training(model.backbone)

        all_names = [n for n, _ in model.backbone.named_modules()]
        def any_in(names, keys): return any(any(k in n for n in names) for k in keys)
        is_clm = hasattr(model.backbone.config, "is_decoder") and model.backbone.config.is_decoder
        task_type = "CAUSAL_LM" if is_clm else "FEATURE_EXTRACTION"

        if any_in(all_names, ["q_proj","k_proj","v_proj","o_proj"]):
            targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]; task_type="CAUSAL_LM"
        elif any_in(all_names, ["W_pack","query_key_value"]):
            cand = []
            if any_in(all_names, ["W_pack"]): cand += ["W_pack","o_proj"]
            if any_in(all_names, ["query_key_value"]): cand += ["query_key_value","dense"]
            targets = list(dict.fromkeys(cand + ["gate_proj","up_proj","down_proj"])); task_type="CAUSAL_LM"
        elif any_in(all_names, ["query","key","value","dense"]):
            targets = ["query","key","value","dense"]; task_type="FEATURE_EXTRACTION"
        else:
            targets = [t.strip() for t in args.lora_target.split(",") if t.strip()]

        lconf = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                           bias="none", task_type=task_type, target_modules=targets)
        model.backbone = get_peft_model(model.backbone, lconf)
        for n,p in model.named_parameters():
            if n.startswith("backbone.") and "lora_" not in n: p.requires_grad_(False)
        try: model.backbone.print_trainable_parameters()
        except Exception: pass

    # optimizer / sched
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],  "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(grouped, lr=args.lr)
    steps = max(1, len(tr)*args.epochs); warm = int(steps*args.warmup_ratio)
    sch = get_linear_schedule_with_warmup(opt, warm, steps)

    # loss
    crit = nn.CrossEntropyLoss()

    best = None
    centers = class_centers
    for ep in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(tr, desc=f"epoch {ep}")
        for b in pbar:
            opt.zero_grad(set_to_none=True)
            logits = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
            y = b["y_cls"].to(dev)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # evaluate (map class -> center numeric)
        model.eval(); ys, yh = [], []
        with torch.no_grad():
            for vb in va:
                lg = model(vb["input_ids"].to(dev), vb["attention_mask"].to(dev))
                pred = lg.argmax(-1).cpu()
                yh.append(pred); ys.append(vb["y_cont"])
        y = torch.cat(ys).float()
        yhat_ids = torch.cat(yh).long()
        centers_t = torch.tensor(centers, dtype=torch.float)
        p = centers_t[yhat_ids]
        rel = (p - y).abs() / y.abs().clamp(min=1e-6)
        m = {
            "pass@1": (rel<=0.33).float().mean().item(),
            "avg_error": (p - y).abs().mean().item(),
            "aggregated_error": (p - y).sum().abs().div(y.abs().sum().clamp(min=1e-6)).item(),
            "cls_acc": (yhat_ids == torch.bucketize(y, boundaries=torch.tensor(bin_edges))).float().mean().item() if False else None,
        }
        print(f"[valid] pass@1={m['pass@1']:.4f} avg={m['avg_error']:.2f} aggr={m['aggregated_error']:.4f}")

        score = m["aggregated_error"]
        if best is None or score < best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "tokenizer": args.model_name,
                    "args": vars(args),
                    "metrics": m,
                    "class_edges": bin_edges,
                    "class_centers": centers,
                    "num_classes": num_classes,
                    "task": "classify"
                },
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

    torch.save(
        {"model": model.state_dict(), "tokenizer": args.model_name, "args": vars(args),
         "class_edges": bin_edges, "class_centers": centers, "num_classes": num_classes, "task":"classify"},
        os.path.join(args.output_dir, "last.pt")
    )

if __name__ == "__main__":
    main()
