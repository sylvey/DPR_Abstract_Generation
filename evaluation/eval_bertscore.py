#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import json # 導入 json 庫用於讀取 jsonl 檔案
import sys
# 導入 AutoTokenizer，用於精確截斷
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="outputs/pairs.jsonl",
                        help="JSONL 檔案路徑，包含 'pred' 和 'ref' 欄位")
    
    parser.add_argument("--ref", default="ref", help="Input JSONL file")
    parser.add_argument("--pred", default="pred", help="Input JSONL file")

    parser.add_argument("--limit", type=int, default=10084,
                        help="限制讀取的紀錄筆數，以匹配 refs.txt 的正確數量")
    parser.add_argument("--model_type", default="../led/models/scibert",
                        help="HF 模型名稱或本機路徑（例如 SciBERT）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=9)
    parser.add_argument("--no_idf", action="store_true", help="關閉 IDF 加權")
    parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if args.device == "cuda" else "cpu"

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
                    preds.append(str(record[args.pred]).rstrip("\n"))
                    refs.append(str(record[args.ref]).rstrip("\n"))
                    count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping malformed line/record in {args.pairs}. Error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: {args.pairs} not found. Please check the file path.")
        return

    if not preds:
        print("Error: No valid prediction data was loaded.")
        return
        
    assert len(preds) == len(refs), "Error: Data extraction from pairs.jsonl failed to match lengths."
    
    print(f"[Loaded] Successfully loaded {len(preds)} pairs from {args.pairs}")

    # =============================================================
    MAX_LENGTH = 510 # 使用 510 留出 [CLS] 和 [SEP] 的位置
    try:
        # 使用 AutoTokenizer 加載您指定的模型 (SciBERT)
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=False)
        
        # 定义一个帮助函數來截斷文本
        def truncate_text(text_list):
            # 1. tokenization (啟用截斷)
            tokens = tokenizer(
                text_list, 
                max_length=MAX_LENGTH, 
                truncation=True, 
                padding=False,
                add_special_tokens=False # 不在這裡加 [CLS]/[SEP]
            )['input_ids']
            
            # 2. decode 回字串 ( bert-score 需要字串輸入)
            # 這裡使用 skip_special_tokens=False 以確保 tokenization 的準確性，雖然 decode 結果會被 BERTScore 重新處理
            return [tokenizer.decode(ids, skip_special_tokens=False) for ids in tokens]

        # 对所有预测和參考進行強制截斷
        preds = truncate_text(preds)
        refs = truncate_text(refs)
        
        print(f"INFO: Successfully truncated all sequences to max token length {MAX_LENGTH} using {args.model_type} tokenizer.")
        
    except Exception as e:
        # 如果無法加載或截斷 (例如 `transformers` 版本太舊), 則輸出警告
        sys.stderr.write(f"WARNING: Manual truncation failed. Proceeding with original long texts. Error: {e}\n")
    # =============================================================
    

    
    from evaluate import load as load_metric
    bertscore = load_metric("bertscore")

    res = bertscore.compute(
        predictions=preds,
        references=refs,
        model_type=args.model_type,
        lang="en",
        idf=not args.no_idf,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        device=device,
    )

    p = float(np.mean(res["precision"]))
    r = float(np.mean(res["recall"]))
    f1 = float(np.mean(res["f1"]))

    print("== BERTScore (SciBERT) ==")
    print(f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}")

if __name__ == "__main__":
    main()