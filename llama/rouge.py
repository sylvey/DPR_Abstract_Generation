#!/usr/bin/env python3
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", default="outputs/preds.txt")
    parser.add_argument("--refs",  default="outputs/refs.txt")
    parser.add_argument("--types", default="rouge1,rouge2,rougeL",
                        help="Comma-separated: rouge1,rouge2,rougeL,rougeLsum")
    args = parser.parse_args()

    from evaluate import load as load_metric
    rouge = load_metric("rouge")

    with open(args.preds, "r", encoding="utf-8") as f:
        preds = [line.rstrip("\n") for line in f]
    with open(args.refs, "r", encoding="utf-8") as f:
        refs = [line.rstrip("\n") for line in f]

    assert len(preds) == len(refs), "preds / refs length mismatch"

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    res = rouge.compute(
        predictions=preds,
        references=refs,
        rouge_types=types,
        use_aggregator=True,
        use_stemmer=True
    )

    print("== ROUGE ==")
    for t in types:
        # evaluate 的 rouge 分數物件通常帶有 mid/fmeasure，可直接印
        score = res[t]
        try:
            print(f"{t.upper():<10} F1={score.mid.fmeasure:.4f} (P={score.mid.precision:.4f}, R={score.mid.recall:.4f})")
        except AttributeError:
            # 某些版本回傳 float
            print(f"{t.upper():<10} {float(score):.4f}")

if __name__ == "__main__":
    main()
