#!/usr/bin/env python3
import argparse, json, numpy as np
from evaluate import load as eval_load

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl_path", help="Path to test_generations_*.jsonl")
    ap.add_argument("--model_type", default="../led/models/scibert",
                    help="Backbone for BERTScore (e.g., roberta-large, or local SciBERT path)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--filter_errors", action="store_true",
                    help="Drop lines whose prediction is [OOM ERROR] or [Runtime ERROR]")
    ap.add_argument("--rescale_with_baseline", action="store_true",
                    help="Use baseline rescaling (usually OFF for SciBERT)")
    args = ap.parse_args()

    preds, refs = [], []
    for rec in read_jsonl(args.jsonl_path):
        p = rec["prediction"]
        r = rec["reference"]
        if args.filter_errors and p in ("[OOM ERROR]", "[Runtime ERROR]"):
            continue
        preds.append(p)
        refs.append(r)
    
     # === 手动为本地模型路径添加层数信息 (SciBERT=12层) ===
    import bert_score.utils
    if args.model_type not in bert_score.utils.model2layers:
        bert_score.utils.model2layers[args.model_type] = 12

    # VVVVVV 强制手動截斷 VVVVVV
    from transformers import AutoTokenizer
    MAX_LENGTH = 510
    try:
        # 使用 AutoTokenizer 加载您指定的模型 (SciBERT)
        # 注意: use_fast=False 可以避免某些 tokenization 的潛在問題
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=False)
        
        # 定义一个帮助函数来截断文本
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
            return [tokenizer.decode(ids, skip_special_tokens=False) for ids in tokens]

        # 对所有预测和參考進行強制截斷
        preds = truncate_text(preds)
        refs = truncate_text(refs)
        
        print(f"INFO: Successfully truncated all sequences to max length {MAX_LENGTH} using {args.model_type} tokenizer.")
        
    except Exception as e:
        # 如果無法加載或截斷 (例如 `transformers` 版本太舊), 則輸出警告
        print(f"WARNING: Manual truncation failed. Proceeding with original long texts. Error: {e}")
    # ^^^^^^ 强制手動截斷 ^^^^^^

    bertscore = eval_load("bertscore")
    bs = bertscore.compute(
        predictions=preds,
        references=refs,
        lang="en",
        model_type=args.model_type,
        rescale_with_baseline=args.rescale_with_baseline,
        device=args.device,
        batch_size=args.batch_size,
        # max_length=512,
    )

    p = float(np.mean(bs["precision"]))
    r = float(np.mean(bs["recall"]))
    f = float(np.mean(bs["f1"]))
    print(f"=== BERTScore ({bs.get('model_type','')}) ===")
    print(f"P: {p:.4f}  R: {r:.4f}  F1: {f:.4f}")

if __name__ == "__main__":
    main()
