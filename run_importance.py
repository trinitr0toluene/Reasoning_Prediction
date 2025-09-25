# -*- coding: utf-8 -*-
"""
Runner script: compute sentence importance (ACC-Δ and/or LL-Δ).

Usage examples:
  python run_importance.py --raw /path/raw_masked.jsonl --sentences /path/sentences.jsonl \
      --out /path/importance_scores.jsonl --cache_dir /path/cache --acc_only

  python run_importance.py --ll_only   # if your adapter implements logprob
"""

import argparse
from pathlib import Path
from sentence_importance import evaluate_dataset
# from sentence_importance import DummyEchoModel  # replace with a real adapter
from hf_adapter import HFGenModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="raw_masked.jsonl")
    ap.add_argument("--sentences", type=str, default="sentences.jsonl")
    ap.add_argument("--out", type=str, default="importance_scores.jsonl")
    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--acc_only", action="store_true")
    ap.add_argument("--ll_only", action="store_true")
    args = ap.parse_args()

    # TODO: switch to a real model adapter, e.g. HFGenModel(...)
    # model = DummyEchoModel()
    model = HFGenModel("Qwen/Qwen2.5-3B-Instruct")

    out_path = evaluate_dataset(
        model=model,
        path_raw=Path(args.raw),
        path_sent=Path(args.sentences),
        out_path=Path(args.out),
        cache_dir=Path(args.cache_dir),
        num_samples=args.num_samples,
        use_acc_delta= (not args.ll_only),
        use_ll_delta=args.ll_only,
    )
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
