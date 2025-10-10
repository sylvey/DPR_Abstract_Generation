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
def generate_answer(batch):
    try:
        with torch.no_grad():
            inputs_dict = tokenizer(
                batch["article"],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs_dict["input_ids"].to(device)
            attention_mask = inputs_dict["attention_mask"].to(device)

            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1  # 讓第一個 token (通常是 BOS) 有 global attention

            predicted_ids = led.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                max_length=512,
                num_beams=4,
            )

            decoded = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            batch["predicted_abstract"] = decoded

    except torch.cuda.OutOfMemoryError:
        print("⚠️ Skipping example due to OOM in generate_answer.")
        torch.cuda.empty_cache()
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
    global tokenizer, led, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache_path = "../data/pubmed_datasets.pkl"

    if os.path.exists(cache_path):
        print(f"✅ Loading datasets from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            pubmed_train, pubmed_val, pubmed_test = pickle.load(f)

    else:
        print("🚀 No cache found. Processing from CSV...")
        df = pd.read_csv('../data/all_articles5-v2.csv')

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

        pubmed_train = Dataset.from_pandas(train_df[["article", "abstract"]].reset_index(drop=True))
        pubmed_val   = Dataset.from_pandas(val_df[["article", "abstract"]].reset_index(drop=True))
        pubmed_test  = Dataset.from_pandas(test_df[["article", "abstract"]].reset_index(drop=True))

        # pubmed_train = pubmed_train.select(range(min(len(pubmed_train), 8000)))
        # pubmed_val   = pubmed_val.select(range(min(len(pubmed_val), 800)))
        # pubmed_test  = pubmed_test.select(range(min(len(pubmed_test), 800)))

        tokenizer = AutoTokenizer.from_pretrained("./led_pubmed_model")
        encoder_max_length = 512
        decoder_max_length = 512

        def process_data_to_model_inputs(example):
            inputs = tokenizer(
                example["article"],
                padding="max_length",
                truncation=True,
                max_length=encoder_max_length,
            )
            outputs = tokenizer(
                example["abstract"],
                padding="max_length",
                truncation=True,
                max_length=decoder_max_length,
            )
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            labels = outputs.input_ids

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "global_attention_mask": [1] + [0] * (len(input_ids) - 1),
                "labels": [-100 if token == tokenizer.pad_token_id else token for token in labels]
            }

        pubmed_train = pubmed_train.map(
            process_data_to_model_inputs,
            batched=False,
            remove_columns=["article", "abstract"],
            load_from_cache_file=False,
            desc="🧪 Mapping training data"
        )
        pubmed_val = pubmed_val.map(
            process_data_to_model_inputs,
            batched=False,
            remove_columns=["article", "abstract"],
            load_from_cache_file=False,
            desc="🧪 Mapping val data"
        )

        # 存快取
        with open(cache_path, "wb") as f:
            pickle.dump((pubmed_train, pubmed_val, pubmed_test), f)
        print(f"💾 Saved processed datasets to {cache_path}")

    print("✅ pubmed_train size:", len(pubmed_train))
    print("✅ pubmed_val size:", len(pubmed_val))
    print("✅ pubmed_test size:", len(pubmed_test))
    print(pubmed_train[0])

    pubmed_train = pubmed_train.flatten_indices()
    pubmed_val = pubmed_val.flatten_indices()
    pubmed_test = pubmed_test.flatten_indices()

    # 再強制重建連續視圖（保險，避免殘留間接索引）
    pubmed_train = pubmed_train.select(range(len(pubmed_train)))
    pubmed_val   = pubmed_val.select(range(len(pubmed_val)))
    pubmed_test   = pubmed_test.select(range(len(pubmed_test)))

    # === 2) 載入 tokenizer / model（arXiv 版）===
    model_dir = "models/led-arxiv"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    led = LEDForConditionalGenerationWithGlobalMask.from_pretrained(
        model_dir,
        use_cache=False,   # 生成時可開可關；關閉對長文較省顯存
        # torch_dtype=torch.float16,  # 若顯存夠、且希望更快可打開
    )
    led.to(device)
    led.eval()

    # 可選：固定生成超參（和你訓練時一致）
    led.config.num_beams = 4
    led.config.max_length = 512
    led.config.min_length = 100
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3

    # === 3) 逐批生成摘要 ===
    result = pubmed_test.map(generate_answer, batched=True, batch_size=1)

    # === 4) 存檔 ===
    out_dir = Path("./models/preds")
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    gen_path = out_dir / f"test_generations_{stamp}.jsonl"
    meta_path = out_dir / f"test_generations_{stamp}.meta.json"

    with gen_path.open("w", encoding="utf-8") as f:
        for i, (p, r) in enumerate(zip(result["predicted_abstract"], result["abstract"])):
            rec = {"id": i, "prediction": p, "reference": r}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "time": stamp,
        "model_name_or_path": "models/led-arxiv",
        "num_beams": getattr(led.config, "num_beams", None),
        "max_length": getattr(led.config, "max_length", None),
        "min_length": getattr(led.config, "min_length", None),
        "length_penalty": getattr(led.config, "length_penalty", None),
        "early_stopping": getattr(led.config, "early_stopping", None),
        "no_repeat_ngram_size": getattr(led.config, "no_repeat_ngram_size", None),
        "tokenizer": tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "unknown",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved generations to: {gen_path}")
    print(f"📝 Saved metadata to:    {meta_path}")

    # === 5) 計算 ROUGE ===
    rouge = load("rouge")
    rouge_result = rouge.compute(
        predictions=result["predicted_abstract"],
        references=result["abstract"],
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_aggregator=True
    )
    print("ROUGE-1:", rouge_result["rouge1"])
    print("ROUGE-2:", rouge_result["rouge2"])
    print("ROUGE-L:", rouge_result["rougeL"])

if __name__ == "__main__":
    main()
