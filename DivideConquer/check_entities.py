import sys
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
import json
from get_umls import GetUMLS
from langchain_ollama import ChatOllama
# import prompts
import concurrent.futures
import pandas as pd
import ast
import os
from tqdm import tqdm
import numpy as np
from data_seperate_column import load
import argparse
from prompts import ParagraphPrompts, AbstractPrompts
import spacy
import pytextrank
from scispacy.linking import EntityLinker

from pydantic import BaseModel, Field, field_validator
from typing import Literal

def batch_check_umls_to_json(target_ids, input_path, output_path=None, top_n=5, use_filtered=2):
    # 1. 初始化工具
    print(f"🔧 Loading SciSpacy model and UMLS linker...")
    nlp = spacy.load("en_core_sci_md")
    nlp.add_pipe("textrank")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, 
        "linker_name": "umls"
    })
    linker = nlp.get_pipe("scispacy_linker")
    umls_tool = GetUMLS(nlp, linker)

    print(f"🚀 Loading dataset from {input_path}...")
    data = load(file_path=input_path, split=False) 
    
    target_set = set(str(tid) for tid in target_ids)
    found_articles = []
    
    for row in data:
        current_id = str(row.get('article_id'))
        if current_id in target_set:
            found_articles.append(row)
            target_set.remove(current_id) # 找到就移除，增加效率
    
    if not found_articles:
        print(f"❌ Error: None of the Article IDs {target_ids} were found.")
        return

    if target_set:
        print(f"⚠️ Warning: Could not find Article IDs: {list(target_set)}")

    all_results = {
        "metadata": {
            "top_n": top_n,
            "use_filtered": use_filtered,
            "total_articles": len(found_articles)
        },
        "articles": {}
    }

    sections = ["background", "objective", "methods", "results", "conclusions", "none"]

    extract_func = umls_tool.get_umls_with_textrank
    if use_filtered == 1:
        extract_func = umls_tool.get_umls_with_textrank
    elif use_filtered == 2:
        extract_func = umls_tool.get_umls_with_original_text
    elif use_filtered == 3:
        extract_func = umls_tool.get_umls_with_original_text_mesh

    # 4. 批次處理
    for article in found_articles:
        aid = str(article.get('article_id'))
        print(f"🔍 Extracting UMLS terms for article: {aid}...")
        
        article_data = {}
        for sec in sections:
            text = article.get(sec)
            if not text or str(text).lower() == 'nan':
                continue
                
            if isinstance(text, list):
                text = " ".join([str(t) for t in text if t])

            terms = extract_func(text, top_n=top_n)
            article_data[sec] = {
                "raw_text_preview": text[:150] + "...",
                "extracted_terms": terms
            }
        
        all_results["articles"][aid] = article_data

    # 5. 寫入 JSON 檔案
    if output_path is None:
        output_path = f"batch_umls_check.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Batch extraction complete! Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 使用 nargs='+' 讓 argparse 接受一個或多個 article_id
    parser.add_argument("--article_ids", type=str, nargs='+', required=True, 
                        help="List of article IDs separated by space (e.g., --article_ids 123 456 789)")
    parser.add_argument("--input_jsonl", default="dataset/all_articles5-v2_sections_combined.parquet")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--top_n", type=int, default=5)
    # parser.add_argument("--filtered", action="store_true")
    parser.add_argument("--umls_filter", default=2)

    
    
    args = parser.parse_args()
    output_json =  f"batch_umls_check_top{args.top_n}_filtered{args.umls_filter}.json" if not args.output_json else args.output_json
    
    batch_check_umls_to_json(
        target_ids=args.article_ids,
        input_path=args.input_jsonl,
        output_path=output_json,
        top_n=args.top_n,
        use_filtered=args.umls_filter
    )