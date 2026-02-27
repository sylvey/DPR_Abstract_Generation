#!/usr/bin/env python3
import argparse
import sys
import json
from evaluate import load as load_metric

def main():
    parser = argparse.ArgumentParser()
    # 移除 --preds 和 --refs，改為一個 --pairs 參數
    parser.add_argument("--pairs", default="outputs/pairs.jsonl",
                        help="JSONL 檔案路徑，包含 'pred' 和 'ref' 欄位")
    parser.add_argument("--ref", default="ref", help="Input JSONL file")
    parser.add_argument("--pred", default="pred", help="Input JSONL file")
    parser.add_argument("--limit", type=int, default=10084,
                        help="限制讀取的紀錄筆數，以匹配 refs.txt 的正確數量")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=9)
    parser.add_argument("--no_idf", action="store_true", help="關閉 IDF 加權")
    parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    args = parser.parse_args()

    # =============================================================
    #           新的資料讀取邏輯：從 pairs.jsonl 載入並限制數量
    # =============================================================
    preds = []
    refs = []
    count = 0

    try:
        with open(args.pairs, "r", encoding="utf-8") as f:
            for line in f:
                if count >= args.limit:
                    break # 達到預設限制，停止讀取
                
                try:
                    record = json.loads(line)
                    # 假設 'pred' 和 'ref' 是紀錄中的正確欄位名稱
                    preds.append(str(record[args.pred]).rstrip("\n"))
                    refs.append(str(record[args.ref]).rstrip("\n"))
                    count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping malformed line/record in {args.pairs}. Error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: {args.pairs} not found. Please check the file path.")
        return

    # 檢查是否讀取到資料
    if not preds:
        print("Error: No valid prediction data was loaded.")
        return
        
    # 檢查長度是否匹配 (在這個點上應該匹配，因為我們是同步讀取)
    assert len(preds) == len(refs), "Error: Data extraction from pairs.jsonl failed to match lengths."
    
    print(f"[Loaded] Successfully loaded {len(preds)} pairs from {args.pairs}")
 
    rouge = load_metric("./rouge/rouge.py")
    types = ["rouge1","rouge2","rougeL"]

    # types = [t.strip() for t in args.types.split(",") if t.strip()]
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
