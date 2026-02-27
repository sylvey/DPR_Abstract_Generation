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

sys.stdout.reconfigure(line_buffering=True)
tqdm.pandas()


from pydantic import BaseModel, Field, field_validator
from typing import Literal

class SummaryOutput(BaseModel):
    summary: str = Field(..., description="The generated biomedical paragraph summary.")
    reasoning: str = Field(..., description="Internal logic for classification and flow.")

    # @field_validator("summary")
    # @classmethod
    # def check_prohibited_terms(cls, v: str) -> str:
    #     forbidden = [ "assistant", "reasoning", "summarize"]
    #     for word in forbidden:
    #         if word in v.lower():
    #             raise ValueError (f"Output contains forbidden word: {word}")
    #     return v

class AbstractOutput(BaseModel):
    abstract: str = Field(..., description="The final generated biomedical abstract.")
    reasoning: str = Field(..., description="Internal logic for classification and flow.")

    # @field_validator("abstract")
    # @classmethod
    # def check_prohibited_terms(cls, v: str) -> str:
    #     forbidden = ["assistant", "reasoning", "refine"]
    #     for word in forbidden:
    #         if word in v.lower():
    #             raise ValueError (f"Output contains forbidden word: {word}")
    #     return v



def generate_paragraph_summary(
        example: str, 
        sec_norm: str, 
        summary_chain, 
        sent_len: str,
        prompt_version: int = 1, 
        compression_ratio: float = 26.2,
        get_umls_terms_textRank=None,
        top_umls_terms: int = 5
        ):
    

    if prompt_version == 1:
        sentence_length = {
            'background': int(sent_len[0]),
            'objective': int(sent_len[1]),
            'methods': int(sent_len[2]),
            'results': int(sent_len[3]),
            'conclusions': int(sent_len[4]),
            'none': int(sent_len[5])
        }
        systemprompt = ParagraphPrompts.prompt1(sec_norm, sentence_length)
        userprompt = f"paragraph text:\n{example}"
    elif prompt_version == 2:
        sentence_length = {
            'background': int(sent_len[0]),
            'objective': int(sent_len[1]),
            'methods': int(sent_len[2]),
            'results': int(sent_len[3]),
            'conclusions': int(sent_len[4]),
            'none': int(sent_len[5])
        }
        systemprompt = ParagraphPrompts.prompt2(sec_norm, sentence_length)
        userprompt = f"paragraph text:\n{example}"
    elif prompt_version == 3:
        systemprompt, userprompt = ParagraphPrompts.prompt3(sec_norm, example, compression_ratio)
    elif prompt_version == 4:
        systemprompt, userprompt = ParagraphPrompts.prompt4(sec_norm, example, compression_ratio)
    elif prompt_version == 5:
        systemprompt, userprompt = ParagraphPrompts.prompt5(sec_norm, example)
    elif prompt_version == 6:
        systemprompt, userprompt = ParagraphPrompts.prompt6(sec_norm, example)
    elif prompt_version == 7:
        systemprompt, userprompt = ParagraphPrompts.prompt7(sec_norm, example, get_umls_terms_textRank, top_umls_terms)
    elif prompt_version == 8:
        systemprompt, userprompt = ParagraphPrompts.prompt8(sec_norm, example)
    elif prompt_version == 9:
        systemprompt, userprompt = ParagraphPrompts.prompt9(sec_norm, example, get_umls_terms_textRank, top_umls_terms)
         

    try:
        messages = [
            ("system", systemprompt),
            ("user", userprompt)
        ]
        
        router_result = summary_chain.invoke(messages)
        
        return {
            "status": "success",
            "results": router_result.summary
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def generate_final_abstract(example: str, abstract_chain, prompt_version: int = 1):

    if prompt_version == 1:
        systemprompt = AbstractPrompts.prompt1()
    elif prompt_version == 2:
        systemprompt = AbstractPrompts.prompt2()

    try:
        messages = [
            ("system", systemprompt),
            ("user", f"abstract draft:\n{example}")
        ]
        
        router_result = abstract_chain.invoke(messages)
        
        if router_result is not None:
            return {
                "status": "success",
                "results": router_result.abstract
            }
        
        print("⚠️ Structured output failed. Retrying with raw LLM...")
        raw_response = llm.invoke(messages)
        
        return {
            "status": "success_raw", 
            "results": raw_response.content 
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def preprocess_text(raw_text):
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
    
    return clean_text







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devide_method", default="sec-header",
                        help="Ollama API base URL")
    parser.add_argument("--model_name", default="gpt-oss:20b",
                        help="Ollama model name")
    parser.add_argument("--input_jsonl", default="dataset/all_articles5-v2_sections_combined.parquet",
                        help="Input JSONL file path")
    parser.add_argument("--sent_len", default="112215",
                        help="number of sentences of each section summary")
    parser.add_argument("--keep_columns_version", type=str, default="v1", choices=["v1", "v2"])
    
    parser.add_argument("--paragraph_prompt_version", type=int, default=1)
    parser.add_argument("--abstract_prompt_version", type=int, default=1)
    parser.add_argument("--compression_ratio", type=float, default=26.2)
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser.add_argument("--split", type=str2bool, default=True)
    parser.add_argument("--top_n_umls", type=int, default=5, help="Number of top UMLS terms to include in the prompt (only for paragraph_prompt_version 7)")
    parser.add_argument("--umls_filtered", type=str2bool, default=False, help="Whether to use TextRank-filtered UMLS terms in the prompt (only for paragraph_prompt_version 7)")
    args = parser.parse_args()

    OUTPUT_JSONL = f"outputs/{args.devide_method}/{args.model_name}_{args.paragraph_prompt_version}_{args.abstract_prompt_version}_{args.umls_filtered}_top{args.top_n_umls}.jsonl"
    
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
    pubmed_test = load(file_path=args.input_jsonl, split=args.split, keep_columns_version=args.keep_columns_version)

    
    with_umls_prompts = [7, 9]
    # for test
    # pubmed_test = pubmed_test.select(range(80))
    
    print(f"📂 Loaded {len(pubmed_test)} test examples.")

    print("🔧 Setting up umls tools...")

    if args.paragraph_prompt_version in with_umls_prompts:
        nlp = spacy.load("en_core_sci_md")
        nlp.add_pipe("textrank")

        nlp.add_pipe("scispacy_linker", config={
            "resolve_abbreviations": True, 
            "linker_name": "umls"
        })

        linker = nlp.get_pipe("scispacy_linker")

        umls_tool = GetUMLS(nlp, linker)
    else:
        umls_tool = None

    def process_single_article(exargs):
        idx, row, umls_tool = exargs

        get_umls_func = None
        if args.paragraph_prompt_version in with_umls_prompts and umls_tool:
            if args.umls_filtered:
                get_umls_func = umls_tool.get_umls_with_original_text
            else:
                get_umls_func = umls_tool.get_umls_with_textrank

        if args.keep_columns_version == "v1":
            section_summaries = {
                "background": "",
                "objective": "",
                "methods": "",
                "results": "",
                "conclusions": "",
                "none": ""
            }
        elif args.keep_columns_version == "v2":
            section_summaries = {
                "first_facet": "",
                "second_facet": "",
                "third_facet": "",
                "fourth_facet": "",
                "fifth_facet": "",
                "sixth_facet": ""
            }

        for sec in section_summaries.keys():
        
            raw_text = row[sec]
            clean_text = ""

            if raw_text is None:
                continue

            clean_text = preprocess_text(raw_text)

            if not clean_text or len(clean_text) < 10 or clean_text.lower() == 'nan':
                continue

            response = generate_paragraph_summary(
                example=clean_text,
                sec_norm=sec,
                summary_chain=summary_chain,
                sent_len = args.sent_len,
                prompt_version=args.paragraph_prompt_version,
                compression_ratio=args.compression_ratio,
                get_umls_terms_textRank=get_umls_func,
                top_umls_terms=args.top_n_umls
            )

            section_summaries[sec] = response.get("results", "") if response["status"] == "success" else ""

        if args.keep_columns_version == "v1":
            valid_summaries = [
                section_summaries[k] for k in ["background", "objective", "methods", "results", "conclusions", "none"]
                if section_summaries[k] 
            ]
        elif args.keep_columns_version == "v2":
            valid_summaries = [
                section_summaries[k] for k in ["first_facet", "second_facet", "third_facet", "fourth_facet", "fifth_facet", "sixth_facet"]
                if section_summaries[k] 
            ]
        
        draft_abstract = "\n".join(valid_summaries)

        if not draft_abstract.strip():
            if args.keep_columns_version == "v1":
                groups = {
                    "intro": ["background", "objective"],
                    "main idea": ["methods", "none"],
                    "result and conclusion": ["results", "conclusions"]
                }   
            elif args.keep_columns_version == "v2":
                groups = {
                    "first": ["first_facet", "second_facet"],
                    "second": ["third_facet", "fourth_facet"],
                    "third": ["fifth_facet", "sixth_facet"]
                }
                
            group_summaries = {}

            for group_name, sections in groups.items():
                combined_text = ""
                for sec in sections:
                    raw_text = row.get(sec, "")
                    clean_text = preprocess_text(raw_text)
                    if clean_text:
                        combined_text += clean_text + " "
                
                combined_text = combined_text.strip()

                if len(combined_text) > 20: 
                    
                    response = generate_paragraph_summary(
                        example=combined_text,
                        sec_norm=group_name, 
                        summary_chain=summary_chain,
                        sent_len=args.sent_len,
                        prompt_version=args.paragraph_prompt_version,
                        compression_ratio=args.compression_ratio,
                        get_umls_terms_textRank=get_umls_func,
                        top_umls_terms=args.top_n_umls
                    )
                    
                    if response["status"] == "success":
                        group_summaries[group_name] = response["results"]
                    else:
                        group_summaries[group_name] = combined_text[:300] 
                else:
                    group_summaries[group_name] = ""

            draft_abstract = "\n".join([v for v in group_summaries.values() if v]).strip()

        if not draft_abstract.strip():
            return {
                "article_id": row['article_id'] if 'article_id' in row else idx,
                "generated_abstract": "",
                "abstract": row['abstract'] if 'abstract' in row else "",
                "status": "skipped",
                "error_message": "Input data was empty or invalid after cleaning"
            }

        abstract_response = generate_final_abstract(
            example=draft_abstract,
            abstract_chain=abstract_chain,
            prompt_version=args.abstract_prompt_version
        )

        ground_truth_abstract = row['abstract'] if 'abstract' in row else ""
        result_entry = {
            "article_id": row['article_id'] if 'article_id' in row else idx, 
            "generated_abstract": abstract_response.get("results", "") if abstract_response["status"] == "success" else "",
            "abstract": ground_truth_abstract,
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
                executor.submit(process_single_article, (i, row, umls_tool)): i 
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


    