import torch
import pandas as pd
from transformers import LEDTokenizer, LEDForConditionalGeneration
from datasets import Dataset
import json
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

MODEL_PATH = "./led_pubmed_finetune/checkpoint-11000" 

DATA_PATH = '../data/all_articles5-v2.csv'

OUTPUT_FILE = "rescue_predictions_11000.jsonl"

GEN_CONFIG = {
    "max_length": 512,
    "min_length": 100,
    "num_beams": 4,
    "no_repeat_ngram_size": 3, 
    "repetition_penalty": 1.2,  
    "early_stopping": True,
    "length_penalty": 2.0,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")


def main():
    print(f"📂 Loading model from: {MODEL_PATH}")
    try:
        tokenizer = LEDTokenizer.from_pretrained(MODEL_PATH)
        model = LEDForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
        model.eval() 
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("READING CSV...")
    df = pd.read_csv(DATA_PATH)
    
    df = df.rename(columns={
            "full_text": "article",
            "abstract": "abstract"
        })
    df = df.dropna(subset=["article", "abstract"])
    df = df[df["article"].str.strip().astype(bool)]
    df = df[df["abstract"].str.strip().astype(bool)]
    df["article"] = df["article"].astype(str)
    df["abstract"] = df["abstract"].astype(str)
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


    results = []

    print("🚀 Starting Inference...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        article = row['article']
        reference = row['abstract']
        
        try:
            inputs = tokenizer(
                article,
                padding="max_length",
                truncation=True,
                max_length=512, 
                return_tensors="pt"
            )
            
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1 

            with torch.no_grad():
                predicted_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    **GEN_CONFIG 
                )

            predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
            results.append({
                "id": idx,
                "input_sample": article[:200] + "...", 
                "prediction": predicted_text,
                "reference": reference
            })

            print(f"\n--- Sample {idx} ---")
            print(f"🤖 Pred: {predicted_text[:150]}...")
            
        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ OOM at index {idx}, skipping.")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Error at index {idx}: {e}")

    print(f"💾 Saving results to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print("✅ Done!")

if __name__ == "__main__":
    main()