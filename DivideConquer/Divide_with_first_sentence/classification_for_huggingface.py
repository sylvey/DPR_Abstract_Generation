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
from datasets import load_dataset
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

# def predict(model, tokenizer, device, paragraph, target_sentence):
#     prompt = construct_prompt(paragraph, target_sentence)
    
#     inputs = tokenizer(
#         prompt, 
#         return_tensors="pt", 
#         max_length=CONFIG.max_length, 
#         truncation=True
#     )
    
#     input_ids = inputs["input_ids"].to(device)
#     attention_mask = inputs["attention_mask"].to(device)

#     with torch.no_grad():
#         probs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
#     probs = probs.cpu().numpy()[0]
    
#     pred_id = np.argmax(probs)
#     pred_label = IDS_TO_LABELS[pred_id]
    
#     result_dict = {label: float(prob) for label, prob in zip(LABELS_TO_IDS.keys(), probs)}
    
#     return pred_label, result_dict

def predict_batch(model, tokenizer, device, paragraphs, target_sentences):

    prompts = [construct_prompt(p, s) for p, s in zip(paragraphs, target_sentences)]
    
    # padding=True 會將 batch 內的序列補齊到相同長度
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        max_length=CONFIG.max_length, 
        truncation=True,
        padding=True
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        # 假設 model 會回傳 [batch_size, num_labels]
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
    # 計算機率 (Softmax)
    probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()
    
    results = []
    for probs in probs_batch:
        pred_id = np.argmax(probs)
        pred_label = IDS_TO_LABELS[pred_id]
        results.append(pred_label)
        
    return results


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
    output_file = "../dataset/pubmed-summarization-stream.jsonl"
    
    try:
        model, tokenizer, device = load_model(model_path)
        
        print("\n=== model loaded. Start testing ===\n")

        print("Loading dataset from Hugging Face...")
        full_test_ds = load_dataset("ccdv/pubmed-summarization", "document", cache_dir="../../../data", split="test")
        
        total_rows = len(full_test_ds)

        processed_count = count_lines(output_file)
        
        if processed_count > 0:
            print(f"Found existing output file with {processed_count} rows.")
            print(f"Resuming from row index {processed_count}...")
            working_ds = full_test_ds.select(range(processed_count, total_rows))
        else:
            print("No existing output file found. Starting from scratch.")
            working_ds = full_test_ds

        if len(working_ds) == 0:
            print("All rows have been processed! Nothing to do.")
            sys.exit(0)
        
        print("Preprocessing columns...")
        
        print(f"Total rows to process: {len(working_ds)}")
        # with open(output_file, 'a', encoding='utf-8') as f_out:

        #     for i, row in enumerate(working_ds):
                

        #         dict_row = {
        #             'background': '',
        #             'objective': '',
        #             'methods': '',
        #             'results': '',
        #             'conclusions': '',
        #             'none': ''
        #         }

                
        #         paragraphs = split_paragraphs_standard(row['article'])

        #         for para in paragraphs:
        #             if not para.strip(): continue
        #             sentences = sent_tokenize(para.strip())
        #             target_sentence = sentences[0] if len(sentences) > 0 else para
        #             label, probs = predict(model, tokenizer, device, para, target_sentence)

        #             dict_row[label] += para + "\n\n"

                

        #         save_data = row.to_dict()
        #         save_data.update(dict_row)
                
        #         clean_data = clean_for_json(save_data)
                
        #         json.dump(clean_data, f_out, ensure_ascii=False)
        #         f_out.write('\n') 
        #         f_out.flush() 
                
        #         current_idx = processed_count + i + 1
        #         if current_idx % 10 == 0:
        #             print(f"Processed {current_idx}/{total_rows} rows...")

        MINI_BATCH_SIZE = 2

        with open(output_file, 'a', encoding='utf-8') as f_out:
            for i, row in enumerate(working_ds):
                dict_row = {k: '' for k in LABELS_TO_IDS.keys()}

                all_paragraphs = split_paragraphs_standard(row['article'])
                if not all_paragraphs:
                    continue
                target_sentences = []
                valid_paragraphs = []
                for p in all_paragraphs:
                    p_clean = p.strip()
                    if not p_clean: continue
                    
                    sentences = sent_tokenize(p_clean)
                    target_sentences.append(sentences[0] if len(sentences) > 0 else p_clean)
                    valid_paragraphs.append(p_clean)

                if valid_paragraphs:
                    all_labels = []
                    
                    for j in range(0, len(valid_paragraphs), MINI_BATCH_SIZE):
                        batch_paras = valid_paragraphs[j : j + MINI_BATCH_SIZE]
                        batch_targets = target_sentences[j : j + MINI_BATCH_SIZE]
                        
                        batch_labels = predict_batch(model, tokenizer, device, batch_paras, batch_targets)
                        all_labels.extend(batch_labels)

                    for label, para in zip(all_labels, valid_paragraphs):
                        if label in dict_row:
                            dict_row[label] += para + "\n\n"
                        else:
                            dict_row['none'] += para + "\n\n"

                save_data = dict(row)
                save_data.update(dict_row)
                clean_data = clean_for_json(save_data)
                
                json.dump(clean_data, f_out, ensure_ascii=False)
                f_out.write('\n') 
                f_out.flush() 
                
                current_idx = processed_count + i + 1
                if current_idx % 10 == 0:
                    print(f"Processed {current_idx}/{total_rows} rows...")
                

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
        
    except FileNotFoundError:
        print(f"Error for model {model_path}")
        
    except Exception as e:
        print(f"Error {e}")
        
    finally:
        print("Exiting...")