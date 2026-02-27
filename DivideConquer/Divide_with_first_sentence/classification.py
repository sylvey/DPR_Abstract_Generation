import torch
from transformers import AutoTokenizer
from argparse import Namespace
import numpy as np
import warnings
import pandas as pd
import ast
import nltk
nltk.download('punkt') 
from nltk.tokenize import sent_tokenize
import re
import pyarrow as pa
import pyarrow.parquet as pq
import signal
import sys
import json
import os
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from model import space_thinking_llm

CONFIG = Namespace(
    model_name_or_path="google/gemma-2b", 
    tokenizer_name_or_path="google/gemma-2b",
    dropout_thinking_linear=0.1,
    num_generate_tokens=2,    
    max_length=512,           
    default_threshold=0.4     
)

LABELS_TO_IDS = {"background": 0, "objective": 1, "methods": 2, "results": 3, "conclusions": 4, "none": 5}
IDS_TO_LABELS = {v: k for k, v in LABELS_TO_IDS.items()}
NUM_LABELS = len(LABELS_TO_IDS)

def load_model(model_path="./best_model.mdl"):
    print(f"loading Base Model: {CONFIG.model_name_or_path} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = space_thinking_llm(CONFIG, num_labels=NUM_LABELS)

    print(f"loading weights: {model_path} ...")
    checkpoint = torch.load(model_path, map_location="cpu") 
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    keys = model.load_state_dict(state_dict, strict=False)
    print(f"Weight Loaded. Missing keys (expected): {len(keys.missing_keys)}")
    
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"
        
    model.eval()
    return model, tokenizer, device

def construct_prompt(paragraph, target_sentence):
    
    prompt = (
        f"The paragraph is \"{paragraph}\". "
        f"Select from rhetorical labels including background, objective, method, result and conclusion, "
        f"the sentence \"{target_sentence}\" plays rhetorical role in the paragraph as "
    )
    return prompt

def predict(model, tokenizer, device, paragraph, target_sentence):
    prompt = construct_prompt(paragraph, target_sentence)
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=CONFIG.max_length, 
        truncation=True
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        probs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
    probs = probs.cpu().numpy()[0]
    
    pred_id = np.argmax(probs)
    pred_label = IDS_TO_LABELS[pred_id]
    
    result_dict = {label: float(prob) for label, prob in zip(LABELS_TO_IDS.keys(), probs)}
    
    return pred_label, result_dict


def split_paragraphs_standard(text):
    if not isinstance(text, str) or not text:
        return []
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]

def clean_for_json(data_dict):
    cleaned = {}
    for k, v in data_dict.items():
        if pd.isna(v): # 如果是 NaN 或 None
            cleaned[k] = "" # 轉成空字串
        elif isinstance(v, (np.int64, np.int32)): # 修正 numpy int 問題
            cleaned[k] = int(v)
        elif isinstance(v, (np.float64, np.float32)): # 修正 numpy float 問題
            cleaned[k] = float(v)
        else:
            cleaned[k] = v
    return cleaned

def count_lines(filename):
    if not os.path.exists(filename):
        return 0
    with open(filename, 'rb') as f:
        return sum(1 for _ in f)

if __name__ == "__main__":

    model_path = "best_model.mdl"
    output_file = "../dataset/classified_articles_stream.jsonl"
    
    try:
        model, tokenizer, device = load_model(model_path)
        
        print("\n=== model loaded. Start testing ===\n")

        print("Reading source CSV...")
        df_full = pd.read_csv("../../data/all_articles5-v2.csv")
        
        df_full = df_full.rename(columns={
            "full_text": "article",
            "abstract": "abstract"
        })
        df_full = df_full.dropna(subset=["article", "abstract"])
        df_full = df_full[df_full["article"].str.strip().astype(bool)]
        df_full = df_full[df_full["abstract"].str.strip().astype(bool)]
        df_full["article"] = df_full["article"].astype(str)
        df_full["abstract"] = df_full["abstract"].astype(str)
        len(df_full)

        train_df, temp_df = train_test_split(df_full, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        df_full = test_df.reset_index(drop=True).copy()

        total_rows = len(df_full) 

        processed_count = count_lines(output_file)
        
        if processed_count > 0:
            print(f"Found existing output file with {processed_count} rows.")
            print(f"Resuming from row index {processed_count}...")
            
            df = df_full.iloc[processed_count:].copy() 
        else:
            print("No existing output file found. Starting from scratch.")
            df = df_full.copy()
        
        del df_full 

        if len(df) == 0:
            print("All rows have been processed! Nothing to do.")
            sys.exit(0)
        
        print("Preprocessing columns...")

        df['section_names'] = df['section_names'].apply(ast.literal_eval)
        df['sections'] = df['sections'].apply(ast.literal_eval)
        
        print(f"Total rows to process: {len(df)}")
        with open(output_file, 'a', encoding='utf-8') as f_out:

            for index, row in df.iterrows():
                
                sections = row['sections']

                dict_row = {
                    'background': '',
                    'objective': '',
                    'methods': '',
                    'results': '',
                    'conclusions': '',
                    'none': ''
                }

                for section in sections:
                    paragraphs = split_paragraphs_standard(section)

                    for para in paragraphs:
                        if not para.strip(): continue
                        sentences = sent_tokenize(para.strip())
                        target_sentence = sentences[0] if len(sentences) > 0 else para
                        label, probs = predict(model, tokenizer, device, para, target_sentence)

                        dict_row[label] += para + "\n\n"

                

                save_data = row.to_dict()
                del save_data['sections']      
                del save_data['section_names']

                save_data.update(dict_row)
                
                clean_data = clean_for_json(save_data)
                
                json.dump(clean_data, f_out, ensure_ascii=False)
                f_out.write('\n') 
                f_out.flush() 
                
                if (index + 1) % 10 == 0:
                    print(f"Processed {index + 1}/{total_rows} rows...")

                

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
        
    except FileNotFoundError:
        print(f"Error for model {model_path}")
        
    except Exception as e:
        print(f"Error {e}")
        
    finally:
        print("Exiting...")