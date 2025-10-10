#!/usr/bin/env python3
import argparse, json
from evaluate import load as eval_load

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl_path", help="Path to test_generations_*.jsonl")
    ap.add_argument("--types", nargs="+", default=["rouge1","rouge2","rougeL"],
                    help="ROUGE types to compute")
    ap.add_argument("--aggregator", choices=["auto","none","mid"], default="auto",
                    help="'auto' uses evaluate default (aggregated F1).")
    args = ap.parse_args()

    preds, refs = [], []
    for rec in read_jsonl(args.jsonl_path):
        preds.append(rec["prediction"])
        refs.append(rec["reference"])

    rouge = eval_load("rouge")
    # evaluate's default aggregator yields scalar F1 per type
    res = rouge.compute(predictions=preds, references=refs, rouge_types=args.types, use_aggregator=True)

    print("=== ROUGE (aggregate F1) ===")
    for k in args.types:
        v = res[k]
        # v can be a float or a object with .mid
        try:
            print(f"{k:>7}: {float(getattr(v,'mid',v).fmeasure if hasattr(v,'mid') else v):.4f}")
        except Exception:
            try:
                print(f"{k:>7}: {float(v):.4f}")
            except Exception:
                print(f"{k:>7}: {v}")

if __name__ == "__main__":
    main()
