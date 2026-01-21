import sys
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
import json
from langchain_ollama import ChatOllama
# import prompts
import concurrent.futures
import pandas as pd
import ast
import os
from tqdm import tqdm
import numpy as np
from data import load
import argparse

sys.stdout.reconfigure(line_buffering=True)
tqdm.pandas()


from pydantic import BaseModel, Field, field_validator
from typing import Literal

class SummaryOutput(BaseModel):
    summary: str = Field(..., description="The generated biomedical paragraph summary.")
    reasoning: str = Field(..., description="Internal logic for classification and flow.")

    @field_validator("summary")
    @classmethod
    def check_prohibited_terms(cls, v: str) -> str:
        forbidden = ["analysis", "assistant", "summary", "reasoning", "summarize"]
        for word in forbidden:
            if word in v.lower():
                raise ValueError (f"Output contains forbidden word: {word}")
        return v

class AbstractOutput(BaseModel):
    abstract: str = Field(..., description="The final generated biomedical abstract.")
    reasoning: str = Field(..., description="Internal logic for classification and flow.")

    @field_validator("abstract")
    @classmethod
    def check_prohibited_terms(cls, v: str) -> str:
        forbidden = ["analysis", "assistant", "summary", "reasoning", "final", "refine"]
        for word in forbidden:
            if word in v.lower():
                raise ValueError (f"Output contains forbidden word: {word}")
        return v



def generate_paragraph_summary(example: str, sec_norm: str, summary_chain):
    sentence_length = {
        'background': 1,
        'objective': 1,
        'methods': 2,
        'results': 2,
        'conclusions': 1,
        'none': 5
    }

    systemprompt = f"""You are a biomedical {sec_norm if sec_norm != "none" else "general"} paragraph summarization assistant. The summary should be {sentence_length[sec_norm]} sentences long."""

    try:
        messages = [
            ("system", systemprompt),
            ("user", f"paragraph text:\n{example}")
        ]
        
        router_result = summary_chain.invoke(messages)
        
        return {
            "status": "success",
            "results": router_result.summary
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def generate_final_abstract(example: str, abstract_chain):

    systemprompt = f"""You are a biomedical abstract refinement assistant. Refine the abstract based on the abstract draft."""

    try:
        messages = [
            ("system", systemprompt),
            ("user", f"abstract draft:\n{example}")
        ]
        
        router_result = abstract_chain.invoke(messages)
        
        return {
            "status": "success",
            "results": router_result.abstract
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devide_method", default="sec-header",
                        help="Ollama API base URL")
    parser.add_argument("--model_name", default="gpt-oss:20b",
                        help="Ollama model name")
    parser.add_argument("--input_jsonl", default="dataset/all_articles5-v2_sections_combined.parquet",
                        help="Input JSONL file path")
    args = parser.parse_args()

    OUTPUT_JSONL = f"outputs/{args.devide_method}/{args.model_name}.jsonl"
    
    output_dir = os.path.dirname(OUTPUT_JSONL)
    if output_dir and not os.path.exists(output_dir):
        print(f"📁 Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    print(f"✅ LLM initialized using {args.model_name}. Preparing pipelines...")
    llm = ChatOllama(
        model=args.model_name, 
        temperature=0.1,
    )
    summary_chain = llm.with_structured_output(SummaryOutput)
    abstract_chain = llm.with_structured_output(AbstractOutput)
    
    # 1. load data
    print("🚀 Loading data...")
    _, _, pubmed_test = load(parquest_path=args.input_jsonl)
    
    # for test
    # pubmed_test = pubmed_test.select(range(80))
    
    print(f"📂 Loaded {len(pubmed_test)} test examples.")

    def process_single_article(args):
        idx, row = args
        
        input_data = row["sections_combined"]

        section_summaries = {
            "background": "",
            "objective": "",
            "methods": "",
            "results": "",
            "conclusions": "",
            "none": ""
        }
        
        for sec in input_data:
            raw_text = input_data[sec]
            clean_text = ""

            if raw_text is None:
                continue

            if isinstance(raw_text, list):
                clean_text = " ".join([str(t) for t in raw_text if t])
                
            elif isinstance(raw_text, str):
                text_str = raw_text.strip()
                if text_str.startswith('[') and text_str.endswith(']'):
                    try:
                        actual_list = ast.literal_eval(text_str)
                        if isinstance(actual_list, list):
                            clean_text = " ".join([str(t) for t in actual_list if t])
                        else:
                            clean_text = text_str
                    except:
                        clean_text = text_str
                else:
                    clean_text = text_str

            if not clean_text or len(clean_text) < 10 or clean_text.lower() == 'nan':
                continue

            response = generate_paragraph_summary(
                example=clean_text,
                sec_norm=sec,
                summary_chain=summary_chain
            )

            section_summaries[sec] = response.get("results", "") if response["status"] == "success" else ""

        valid_summaries = [
            section_summaries[k] for k in ["background", "objective", "methods", "results", "conclusions", "none"]
            if section_summaries[k] 
        ]
        
        draft_abstract = "\n".join(valid_summaries)

        if not draft_abstract.strip():
            return {
                "article_id": row['article_id'],
                "generated_abstract": "",
                "abstract": row.get("abstract", ""),
                "status": "skipped",
                "error_message": "Input data was empty or invalid after cleaning"
            }

        abstract_response = generate_final_abstract(
            example=draft_abstract,
            abstract_chain=abstract_chain
        )

        result_entry = {
            "article_id": row['article_id'],  
            "generated_abstract": abstract_response.get("results", "") if abstract_response["status"] == "success" else "",
            "abstract": row.get("abstract", ""),
            "status": abstract_response["status"],
            "error_message": abstract_response.get("message", None)
        }
        return result_entry
    
    MAX_WORKERS = 2 
    
    print(f"⚡ Starting generation with {MAX_WORKERS} workers...")
    print(f"💾 Saving to {OUTPUT_JSONL}...")

    results = []
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(process_single_article, (i, row)): i 
                for i, row in enumerate(pubmed_test)
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(pubmed_test)):
                try:
                    result = future.result()
                    
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush() 
                    
                except Exception as exc:
                    print(f"\n❌ Exception for an article: {exc}")

    print(f"\n✅ Processing complete. Data saved to {OUTPUT_JSONL}")


    