import torch
from transformers import AutoTokenizer, LongT5ForConditionalGeneration

#!/usr/bin/env python3
import os
import json, time
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from evaluate import load
from transformers import AutoTokenizer
from transformers.models.led.modeling_led import LEDForConditionalGeneration
import pickle
import argparse
import re
from datasets import load_dataset

# -----------------------------
# Runtime safety tweaks
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 一次性的 softmax 穩定化（避免 fp16/數值問題）
if not hasattr(F, "_orig_softmax"):
    F._orig_softmax = F.softmax

@torch.no_grad()
def stable_softmax(input, dim=None, **kwargs):
    x = input.to(torch.float32)
    x = torch.nan_to_num(x, nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf"))
    out = F._orig_softmax(x, dim=dim, **kwargs)
    return out.to(input.dtype)

F.softmax = stable_softmax

# -----------------------------
# Inference helpers
# -----------------------------
def generate_answer(batch, output_path):
    try:
        with torch.no_grad():

            inputs = tokenizer(
                batch["article"], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=8192  
            ).to(device)

            sequences = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                num_beams=4,
                early_stopping=True,   
                no_repeat_ngram_size=3,
                use_cache=True       
            )

            # 如果 model.generate 回傳的是物件 (因為 return_dict_in_generate=True)
            if hasattr(sequences, "sequences"):
                sequences = sequences.sequences

            decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            batch["predicted_abstract"] = decoded

            with open(output_path, "a", encoding="utf-8") as f:
                ids = batch.get("article_id", [None]*len(decoded)) 
                
                for idx, pred, ref in zip(ids, decoded, batch["abstract"]):
                    rec = {"article_id": idx, "prediction": pred, "reference": ref}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    except torch.cuda.OutOfMemoryError:
        print("⚠️ Skipping example due to OOM in generate_answer.")
        torch.cuda.empty_cache()
        with open(output_path, "a", encoding="utf-8") as f:
            for _ in batch["article"]:
                 f.write(json.dumps({"error": "OOM"}, ensure_ascii=False) + "\n")
        batch["predicted_abstract"] = ["[OOM ERROR]"] * len(batch["article"])

    except RuntimeError as e:
        print(f"⚠️ RuntimeError in generate_answer: {e}")
        torch.cuda.empty_cache()
        batch["predicted_abstract"] = ["[Runtime ERROR]"] * len(batch["article"])

    return batch

class LEDForConditionalGenerationWithGlobalMask(LEDForConditionalGeneration):
    def forward(self, *args, **kwargs):
        if "global_attention_mask" not in kwargs and "attention_mask" in kwargs:
            attn = kwargs["attention_mask"]
            kwargs["global_attention_mask"] = torch.zeros_like(attn)
            kwargs["global_attention_mask"][:, 0] = 1
        return super().forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs.pop("labels", None)
        return super().generate(*args, **kwargs)

# -----------------------------
# Main (TEST ONLY)
# -----------------------------
def main():
    global tokenizer, model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/led-arxiv")
    parser.add_argument("--input_type", type=str, default="csv")
    parser.add_argument("--input_csv", type=str, default="../data/all_articles5-v2.csv")
    args = parser.parse_args()
    
    if args.input_type == "csv":
        df = pd.read_csv(args.input_csv)

        # rename + clean
        df = df.rename(columns={
            "full_text": "article",
            "abstract": "abstract"
        })
        df = df.dropna(subset=["article", "abstract"])
        df = df[df["article"].str.strip().astype(bool)]
        df = df[df["abstract"].str.strip().astype(bool)]
        df["article"] = df["article"].astype(str)
        df["abstract"] = df["abstract"].astype(str)

        # split
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        pubmed_test  = Dataset.from_pandas(test_df[["article", "abstract", "article_id"]].reset_index(drop=True))

        
        # tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        
        print("✅ pubmed_test size:", len(pubmed_test))
        pubmed_test = pubmed_test.flatten_indices()
    elif args.input_type == "dataset":
        print("Loading dataset from Hugging Face...")
        pubmed_test = load_dataset("ccdv/pubmed-summarization", "document", cache_dir="../data", split="test")

    pubmed_test   = pubmed_test.select(range(len(pubmed_test)))

    # === 2) 載入 tokenizer / model（arXiv 版）===
    model_dir = args.model_dir
    model = LongT5ForConditionalGeneration.from_pretrained(model_dir, return_dict_in_generate=True, torch_dtype=torch.float16,).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    out_dir = Path("./models/preds")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = os.path.basename(args.model_dir)
    gen_path = out_dir / f"test_generations_{stamp}_{name}.jsonl"
    meta_path = out_dir / f"test_generations_{stamp}_{name}.meta.json"

    print(f"📂 Output will be streamed to: {gen_path}")
    
    with open(gen_path, 'w', encoding='utf-8') as f:
        pass

    result = pubmed_test.map(generate_answer, batched=True, batch_size=1, fn_kwargs={"output_path": gen_path})


    meta = {
        "time": stamp,
        "model_name_or_path": args.model_dir,
        "num_beams": getattr(model.config, "num_beams", 4),
        "max_length": getattr(model.config, "max_length", None),
        "tokenizer": tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "unknown",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved generations to: {gen_path}")
    print(f"📝 Saved metadata to:    {meta_path}")


if __name__ == "__main__":
    main()





