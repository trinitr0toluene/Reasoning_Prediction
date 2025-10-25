# -*- coding: utf-8 -*-
"""
eval.py
-------
Evaluate either a regression ckpt from fine_tune.py or a classification ckpt from ft_classify.py,
on the SAME test jsonl. Outputs pass@1 / avg_error / aggregated_error (+ cls_acc if available).

Usage:
python eval.py \
  --ckpt runs/io_only_20k_seed42/best.pt \
  --data-file /root/autodl-fs/openr1_math_test.jsonl \
  --batch-size 8 --max-length 512
"""
import os, json, argparse
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModel, AutoModelForCausalLM
from typing import Any, Dict, List, Tuple
import hashlib, re

# ---------- reuse minimal pieces from fine_tune ----------
def read_jsonl(path: str):
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln: yield json.loads(ln)

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

def best_join_keys(d: Dict[str, Any]) -> List[Tuple]:
    keys = []
    for name in ("uuid","sample_id","id","qid"):
        v = d.get(name); gi = d.get("gen_index")
        if name in ("uuid","sample_id") and gi is not None:
            try: gi_int = int(gi); keys.append((f"{name}+gen", str(v), gi_int))
            except Exception: pass
    for name in ("uuid","sample_id","id","qid"):
        v = d.get(name)
        if v is not None: keys.append((name, str(v)))
    q = d.get("prompt") or d.get("question") or ""
    keys.append(("hash", hashlib.md5(q.encode("utf-8")).hexdigest()))
    return keys

def extract_summary_text(d: Dict[str, Any]) -> str:
    for k in ("summary","summary_text","summary_answer_driven"):
        v = d.get(k)
        if isinstance(v, str) and v.strip(): return v.strip()
    return ""

def build_summary_index(path: str):
    idx = {}; 
    if not path: return idx
    for row in read_jsonl(path):
        s = extract_summary_text(row)
        if not s: continue
        for k in best_join_keys(row):
            if k not in idx: idx[k] = s
    return idx

def _safe_float(y):
    try: return float(str(y).strip())
    except Exception: return None

class RLPDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer, max_length=1024, use_summary=False, summary_index=None):
        self.records = records; self.tok = tokenizer; self.max_length=max_length
        self.use_summary = use_summary; self.sidx = summary_index or {}
    def __len__(self): return len(self.records)
    def _pick_answer(self, r):
        if "final_answer" in r and isinstance(r["final_answer"], str): return strip_think_keep_visible(r["final_answer"])
        if "answer" in r and isinstance(r["answer"], str): return strip_think_keep_visible(r["answer"])
        out = r.get("output", ""); return strip_think_keep_visible(out if isinstance(out,str) else str(out))
    def _lookup_summary(self, r):
        for k in ("summary","summary_text","summary_answer_driven"):
            v = r.get(k)
            if isinstance(v, str) and v.strip(): return v.strip()
        if not self.sidx: return ""
        for k in best_join_keys(r):
            if k in self.sidx: return self.sidx[k]
        return ""
    def __getitem__(self, i):
        r = self.records[i]
        prompt = r.get("prompt", r.get("question","")) or ""
        answer = self._pick_answer(r)
        s_txt = self._lookup_summary(r) if self.use_summary else ""
        if self.use_summary and s_txt:
            x = f"Prompt:\n{prompt}\n\nAnswer:\n{answer}\n\nSummary:\n{s_txt}"
        else:
            x = f"Prompt:\n{prompt}\n\nAnswer:\n{answer}"
        y = r.get("reasoning_token_count", None)
        if y is None:
            for k in ("reasoning_tokens","reasoning_len","hidden_tokens","hidden_len"):
                if k in r and r[k] is not None:
                    y = r[k]; break
        y = _safe_float(y)
        if y is None: y = 0.0
        enc = self.tok(x, truncation=True, max_length=self.max_length, padding=False, return_tensors="pt")
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item["y"] = float(y)
        return item

class MeanPooler(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        s = (last_hidden_state * mask).sum(dim=1); d = mask.sum(dim=1).clamp(min=1e-6)
        return s / d

class RLPRegressor(nn.Module):
    def __init__(self, model_name: str, **fp_kwargs):
        super().__init__()
        try: self.backbone = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **fp_kwargs)
        except Exception: self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True, **fp_kwargs)
        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            for k in ("n_embd","d_model"):
                if hasattr(self.backbone.config, k):
                    v = getattr(self.backbone.config, k); hidden = v[-1] if isinstance(v,(list,tuple)) else int(v); break
        if hidden is None: raise RuntimeError("Cannot infer hidden size.")
        self.pool = MeanPooler()
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
    def _unwrap_base(self):
        bb = self.backbone
        if hasattr(bb,"get_base_model"):
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
            last_hidden = out.last_hidden_state if getattr(out,"last_hidden_state",None) is not None else out.hidden_states[-1]
        z = self.pool(last_hidden, attention_mask)
        return self.head(z).squeeze(-1)

class RLPClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, **fp_kwargs):
        super().__init__()
        try: self.backbone = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **fp_kwargs)
        except Exception: self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True, **fp_kwargs)
        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            for k in ("n_embd","d_model"):
                if hasattr(self.backbone.config, k):
                    v = getattr(self.backbone.config, k); hidden = v[-1] if isinstance(v,(list,tuple)) else int(v); break
        if hidden is None: raise RuntimeError("Cannot infer hidden size.")
        self.pool = MeanPooler()
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, num_classes))
    def _unwrap_base(self):
        bb = self.backbone
        if hasattr(bb,"get_base_model"):
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
            last_hidden = out.last_hidden_state if getattr(out,"last_hidden_state",None) is not None else out.hidden_states[-1]
        z = self.pool(last_hidden, attention_mask)
        return self.head(z)

@torch.no_grad()
def evaluate_reg(model, loader, device, rel_thr=0.33):
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

@torch.no_grad()
def evaluate_cls(model, loader, device, class_centers: List[float], rel_thr=0.33):
    model.eval(); ys, yh_ids = [], []
    for b in loader:
        lg = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        pred = lg.argmax(-1).cpu()
        yh_ids.append(pred); ys.append(b["y"])
    y = torch.cat(ys).float(); yhat_ids = torch.cat(yh_ids).long()
    centers = torch.tensor(class_centers, dtype=torch.float)
    p = centers[yhat_ids]
    rel = (p - y).abs() / y.abs().clamp(min=1e-6)
    cls_acc = None  # 可选：若你传入 true class ids，也可算准确率
    return {
        "pass@1": (rel<=rel_thr).float().mean().item(),
        "avg_error": (p - y).abs().mean().item(),
        "aggregated_error": (p - y).sum().abs().div(y.abs().sum().clamp(min=1e-6)).item(),
        "cls_acc": cls_acc,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--summary-file", type=str, default="")  # 评估通常不拼接 summary，这里留接口
    ap.add_argument("--use-summary", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cargs = ckpt.get("args", {})
    model_name = ckpt.get("tokenizer") or cargs.get("model_name")

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True,
                                        local_files_only=os.path.isdir(model_name))
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    # dataset
    recs = list(read_jsonl(args.data_file))
    sidx = build_summary_index(args.summary_file) if args.use_summary else {}
    ds = RLPDataset(recs, tok, max_length=args.max_length, use_summary=args.use_summary, summary_index=sidx)

    base_collate = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
    def collate_with_y(features):
        ys = torch.tensor([float(f.pop("y")) for f in features], dtype=torch.float)
        batch = base_collate(features); batch["y"] = ys; return batch
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_with_y)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model & (re)apply LoRA if used
    use_lora = bool(cargs.get("use_lora", False))
    is_classify = (ckpt.get("task") == "classify") or ("num_classes" in ckpt)
    if is_classify:
        num_classes = int(ckpt.get("num_classes") or len(ckpt["class_edges"]))
        model = RLPClassifier(model_name, num_classes=num_classes).to(dev)
    else:
        model = RLPRegressor(model_name).to(dev)

    if use_lora:
        from peft import LoraConfig, get_peft_model
        # 目标模块：优先用 ckpt 的配置，其次自动探测
        lora_target = cargs.get("lora_target", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
        all_names = [n for n,_ in model.backbone.named_modules()]
        def any_in(names, keys): return any(any(k in n for n in names) for k in keys)
        is_clm = hasattr(model.backbone.config, "is_decoder") and model.backbone.config.is_decoder
        task_type = "CAUSAL_LM" if is_clm else "FEATURE_EXTRACTION"
        # 若 ckpt 提供 target 就用它，否则自动探测
        targets = [t.strip() for t in lora_target.split(",") if t.strip()]
        if not targets:
            if any_in(all_names, ["q_proj","k_proj","v_proj","o_proj"]):
                targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]; task_type="CAUSAL_LM"
            elif any_in(all_names, ["W_pack","query_key_value"]):
                cand=[]; 
                if any_in(all_names, ["W_pack"]): cand+=["W_pack","o_proj"]
                if any_in(all_names, ["query_key_value"]): cand+=["query_key_value","dense"]
                targets = list(dict.fromkeys(cand+["gate_proj","up_proj","down_proj"])); task_type="CAUSAL_LM"
            elif any_in(all_names, ["query","key","value","dense"]):
                targets=["query","key","value","dense"]; task_type="FEATURE_EXTRACTION"
            else:
                targets=["q_proj","k_proj","v_proj","o_proj"]
        lconf = LoraConfig(
            r=int(cargs.get("lora_r",16)),
            lora_alpha=float(cargs.get("lora_alpha",32)),
            lora_dropout=float(cargs.get("lora_dropout",0.05)),
            bias="none",
            task_type=task_type,
            target_modules=targets,
        )
        model.backbone = get_peft_model(model.backbone, lconf)

    # load weights
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing: print("[eval] missing keys:", len(missing))
    if unexpected: print("[eval] unexpected keys:", len(unexpected))
    model.eval()

    # evaluate
    if is_classify:
        centers = ckpt.get("class_centers")
        if centers is None:
            # fallback: derive centers from edges
            edges = ckpt.get("class_edges")
            if edges is None: raise ValueError("Classification ckpt missing class metadata.")
            centers = []
            low = 0.0
            for e in edges:
                centers.append((low + float(e))/2.0); low = float(e)
        metrics = evaluate_cls(model, dl, dev, class_centers=centers)
    else:
        metrics = evaluate_reg(model, dl, dev)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()


# # ==== 结果统计 ====
# if total_count > 0:
#     mean_error = sum(errors) / len(errors)
#     mean_ratio = sum(error_ratios) / len(error_ratios)
#     accuracy = accurate_count / total_count
#     print(f"\n[Greedy Search] Average numeric error (all cases): {mean_error:.4f}")
#     print(f"[Greedy Search] Average relative error (all cases): {mean_ratio * 100:.2f}%")
#     print(f"[Greedy Search] Accuracy within 25% error margin: {accuracy * 100:.2f}%")
# else:
#     print("No valid numeric predictions found.")
