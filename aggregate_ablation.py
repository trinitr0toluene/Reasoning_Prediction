
import json
import argparse
from collections import defaultdict

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, items):
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=str, required=True, help="ablation_results.jsonl")
    ap.add_argument("--out", type=str, required=True, help="per_step_importance.jsonl")
    args = ap.parse_args()

    # Group records by (uuid, gen_index)
    groups = defaultdict(list)
    for rec in read_jsonl(args.inp):
        groups[(rec["uuid"], rec["gen_index"])].append(rec)

    out_records = []
    for key, recs in groups.items():
        # find base record
        base = next((r for r in recs if r["variant"] == "base"), None)
        if base is None or base.get("is_match") is None:
            # cannot compute drop if no gold or no base
            continue
        base_match = base["is_match"]

        # per-step drops
        for r in recs:
            if r["variant"].startswith("drop_"):
                step_idx = r.get("step_index", -1)
                abl_match = r.get("is_match")
                if abl_match is None:
                    continue
                imp_bin = int(base_match) - int(abl_match)  # 1 means deletion caused correct->incorrect
                out_records.append({
                    "uuid": key[0],
                    "gen_index": key[1],
                    "step_index": step_idx,
                    "imp_bin": imp_bin,
                    "base_is_match": base_match,
                    "abl_is_match": abl_match,
                    "granularity": r.get("granularity"),
                    "model": r.get("model"),
                })

    write_jsonl(args.out, out_records)
    print(f"Wrote {len(out_records)} step-importance records to {args.out}")

if __name__ == "__main__":
    main()
