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

sys.stdout.reconfigure(line_buffering=True)
tqdm.pandas()

MODEL_NAME = "gpt-oss:20b"
INPUT_PARQUET = '/jet/home/slin23/tmp_ondemand_ocean_cis230089p_symlink/slin23/full_text_label/section_normalization/for_section_normalization.parquet'
OUTPUT_JSONL = "ollama_section_labels_streaming.jsonl" 

llm = ChatOllama(
    model=MODEL_NAME, 
    temperature=0.1,
)


print(f"✅ LLM initialized using {MODEL_NAME}. Preparing pipelines...")

# --- Stage 1: Router Schema ---
class RouterOutput(BaseModel):
    '''Classify the **dominant function** of a paragraph into one of 5 broad groups.'''
    group: List[Literal[
        "introduction", 
        "methods", 
        "results", 
        "discussion", 
        "metadata"
    ]] = Field(
        ..., 
        description='''Select one or MORE groups that apply to this paragraph:
- **introduction**: Background, objectives, hypothesis. 
- **methods**: Study design, samples, procedures, statistical methods. 
- **results**: Findings, numerical data, tables descriptions. (Typical Results) 
- **discussion**: Interpretation, implication, limitation, recommendation, future work, conclusion.
- **metadata**: Ethics, funding, availability, authorship, references. '''
    )
    reasoning: str = Field(..., description="Brief reason for this classification.")

# Group A: Introduction
class IntroSolver(BaseModel):
    """Annotate the **Introduction** sections of biomedical papers."""
    labels: List[Literal["background", 
                   "objective", 
                   "none"]] = Field(
        ..., 
        description=(
            '''Select ALL applicable labels:
    - **background**: Sentences that introduce broader context, summarize prior work, or highlight the knowledge gap.
    - **objective**: Sentences that explicitly state the study's aim, purpose, or research question (e.g., "the aim of this study was...", "we sought to...").
    - **none**: If the paragraph strictly does not fit the above (e.g., unrelated text).'''
        )
    )
    reasoning: str = Field(...)

# Group B: Methods
class MethodSolver(BaseModel):
    """Annotate the **Methods** sections of biomedical papers."""
    labels: List[Literal[
        "study_samples",
        "procedure",
        "prior_work_entity",
        "statistical_method",
        "none",
    ]] = Field(
        ..., 
        description=(
            '''Select ALL applicable labels:
    - **study_samples**: Defining and selecting study subjects, patients, animals, or datasets. Includes inclusion/exclusion criteria and recruitment. 
    - **procedure**: How the study was carried out (interventions, materials, measurements, analytic steps).  
    - **prior_work_entity**: Explicit mention of using *existing* entities from previous research (established datasets, corpora, ontologies, benchmarks). 
    - **statistical_method**: Description of statistical tests used (p-values, chi-square) or handling of missing data. 
    - **none**: If the paragraph strictly does not fit the above (e.g., unrelated text).'''
        )
    )
    reasoning: str = Field(...)

# Group C: Results
class ResultSolver(BaseModel):
    """Annotate the **Results** sections of biomedical papers."""
    labels: List[Literal[
        "results_findings",
        "data_statistics",
        "novel_entity",
        "none",
        ]] = Field(
        ..., 
        description=(
            '''Select ALL applicable labels:
    - **results_findings**: Outcomes, numerical results, statistical associations, or effect sizes presented *without* extended interpretation. 
    - **data_statistics**: Baseline or descriptive statistics of the *final study sample* (e.g., demographics, clinical characteristics, group assignments, intervention exposure, or case distributions in the target condition). 
    - **novel_entity**: Introduction of a *newly developed* element (new dataset, new tool, new algorithm) created *by this specific study*. 
    - **none**: If the paragraph strictly does not fit the above.'''
        )
    )
    reasoning: str = Field(...)

# Group D: Discussion
class DiscussionSolver(BaseModel):
    """Annotate the **Discussion** sections of biomedical papers."""
    labels: List[Literal[
        "interpretation",
        "implication",
        "limitation",
        "recommendation",
        "future_work",
        "conclusion",
        "none",
    ]] = Field(
        ..., 
        description=(
            '''Select ALL applicable labels:
    - **interpretation**: Direct explanation of results (mechanism, cause) with certainty. Factual tone. 
    - **implication**: Speculative meaning. Tentative tone (may, might, suggest). Distinct from interpretation and recommendation. 
    - **limitation**: Shortcomings, constraints, missing measurements, weak generalizability. 
    - **recommendation**: Explicit proposal for action/policy (e.g., "clinicians should"). Prescriptive. 
    - **future_work**: Planned future studies, unresolved questions, extensions of current work, or new hypotheses to be tested. 
    - **conclusion**: Summary of main takeaways or contribution. 
    - **none**: If the paragraph strictly does not fit the above (e.g., unrelated text).'''
        )
    )
    reasoning: str = Field(...)

# Group E: Meta Info
class MetaSolver(BaseModel):
    """Annotate the **Meta** sections of biomedical papers."""
    labels: List[Literal[
        "ethics",
        "funding",
        "coi",
        "materials_availability",
        "authorship",
        "acknowledgements",
        "references",
        "none",
    ]] = Field(
        ..., 
        description=(
            '''Select ALL applicable labels:
    - **ethics**: Ethical considerations, IRB approval, consent forms, compliance with guidelines (e.g., Helsinki).
    - **funding**: Financial support acknowledgements (grants, institutions).
    - **coi**: Conflict of interest statements (financial relationships, consultancies).
    - **materials_availability**: Availability of data, code, or materials (repository links, DOIs).
    - **authorship**: Contributions of individual authors.
    - **acknowledgements**: Thanking individuals/orgs who are *not* authors.
    - **references**: Bibliographic entries.
    - **none**: If the paragraph strictly does not fit the above.'''
        )
    )
    reasoning: str = Field(...)

# ==========================================
# 3. 建立 LangChain 結構化模型
# ==========================================
# 這一步會把 Pydantic schema 綁定到 LLM
router_chain = llm.with_structured_output(RouterOutput)

solvers = {
    "introduction": llm.with_structured_output(IntroSolver),
    "methods": llm.with_structured_output(MethodSolver),
    "results": llm.with_structured_output(ResultSolver),
    "discussion": llm.with_structured_output(DiscussionSolver),
    "metadata": llm.with_structured_output(MetaSolver),
}

# ==========================================
# 4. 主邏輯：Two-Stage Pipeline
# ==========================================
def classify_paragraph(text: str):
    """
    執行多標籤分類：Router (Multiple Groups) -> Multiple Solvers
    """
    # print(f"\n--- Processing Paragraph (Length: {len(text)}) ---")
    
    try:
        # Step 1: Route (可能回傳多個 Group)
        router_result = router_chain.invoke(f"Classify all applicable functions of this biomedical paragraph: {text}")
        groups = router_result.group
        # print(f"Step 1 (Router): {groups}")
        
        # Step 2: Solve (針對每個 Group 分別跑 Solver)
        final_results = []
        
        for group in groups:
            if group in solvers:
                # print(f"  -> Running solver for: {group}")
                solver_chain = solvers[group]
                # 呼叫對應的 Solver
                res = solver_chain.invoke(f"Identify all applicable labels for {group} in this paragraph: {text}")
                
                # 過濾掉 'none' 的標籤，只保留有意義的
                valid_labels = [L for L in res.labels if L != "none"]
                
                if valid_labels:
                    final_results.append({
                        "group": group,
                        "labels": valid_labels,
                        "reasoning": res.reasoning
                    })
        
        return {
            "status": "success",
            "results": final_results # 回傳一個 List 包含所有找到的標籤
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def sanitize_for_json(obj):
    """
    [關鍵修正] 遞迴處理資料。
    注意：List/Dict 檢查必須在 pd.isna 之前，否則會報錯。
    """
    # 1. Numpy 轉 Python 原生
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    
    # 2. 容器類 (List/Dict) - 必須先檢查！
    if isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    
    # 3. 最後才檢查是否為 NaN (此時 obj 已經不是 list 了)
    if pd.isna(obj):
        return None
        
    return obj
    

def process_row_and_save(row):
    """
    處理單篇文章並立即寫入 JSONL
    """
    article_id = row['article_id']
    article_sections = row['sections']
    meta_data = {
        "title": row.get('title'),
        "abstract": row.get('abstract'),
        "section_names": row.get('section_names'),
    }
    
    # --- 處理邏輯開始 ---
    article_labels_structure = []
    
    # 資料驗證
    valid_input = True
    if article_sections is None: valid_input = False
    if hasattr(article_sections, 'tolist'): article_sections = article_sections.tolist()
    if not isinstance(article_sections, list): valid_input = False

    if not valid_input:
        # 即使資料有問題，也要記錄一個空結果，避免下次重複跑
        raw_record = {
            "article_id": article_id,
            **meta_data,
            "sections": [],
            "section_labels": [],
            "status": "invalid_input"
        }
    else:
        # 平行處理段落
        MAX_WORKERS = 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for section_paragraphs in article_sections:
                if hasattr(section_paragraphs, 'tolist'): section_paragraphs = section_paragraphs.tolist()
                
                if not isinstance(section_paragraphs, list):
                    article_labels_structure.append([])
                    continue

                future_to_index = {}
                for idx, paragraph in enumerate(section_paragraphs):
                    if not isinstance(paragraph, str) or not paragraph.strip():
                        continue
                    future = executor.submit(classify_paragraph, paragraph)
                    future_to_index[future] = idx

                results_ordered = [[] for _ in range(len(section_paragraphs))]

                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        res = future.result()
                        paragraph_labels = []
                        if res.get("status") == "success":
                            results = res.get("results", [])
                            for item in results:
                                paragraph_labels.extend(item['labels'])
                            paragraph_labels = sorted(list(set(paragraph_labels)))
                        results_ordered[idx] = paragraph_labels
                        print(".", end="", flush=True)
                    except Exception:
                        results_ordered[idx] = []

                article_labels_structure.append(results_ordered)

            print(f"\n✅ Finished Article {article_id}", flush=True)

        
        raw_record = {
            "article_id": article_id,
            **meta_data, # 展開 metadata
            "sections": article_sections,
            "section_labels": article_labels_structure,
            "status": "success"
        }

    # --- 關鍵改動：立即寫入檔案 (Append Mode) ---
    try:
        print(f"💾", end="", flush=True) # 表示開始寫檔

        clean_record = sanitize_for_json(raw_record)

        with open(OUTPUT_JSONL, 'a', encoding='utf-8') as f:
            f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            
        print(f"✅", flush=True)
            
    except Exception as e:
        print(f"\n❌ Save Error {article_id}: {e}", flush=True)

    return article_labels_structure # 回傳給 Pandas (雖然我們主要依賴 JSONL)


if __name__ == "__main__":

    print(f"📂 Loading data from {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    total_articles = len(df)
    print(f"   Original count: {total_articles}")

    # --- 斷點續傳邏輯 ---
    processed_ids = set()
    if os.path.exists(OUTPUT_JSONL):
        print(f"🔄 Found existing output file: {OUTPUT_JSONL}. Checking progress...")
        try:
            with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_ids.add(data['article_id'])
                    except json.JSONDecodeError:
                        continue # 跳過壞掉的行
        except Exception as e:
            print(f"⚠️ Warning reading existing file: {e}")
        
        print(f"   Already processed: {len(processed_ids)} articles.")
    
    if 'article_id' in df.columns:
        df_to_process = df[~df['article_id'].isin(processed_ids)].copy()
    else:
        print("⚠️ 'article_id' column not found! Cannot perform resume logic. Processing ALL.")
        df_to_process = df

    # print("⚠️ TEST MODE: Only processing the first 1 article.")
    # df_to_process = df_to_process.head(2)

    remaining_count = len(df_to_process)
    print(f"🚀 Starting inference on remaining {remaining_count} articles...")


    if remaining_count > 0:
        df_to_process.progress_apply(process_row_and_save, axis=1)
        
        print("\n✅ Batch Processing Complete!")
    else:
        print("\n🎉 All articles have been processed!")
    
    

    # ==========================================
    # 4. (選用) 最後轉回 Parquet
    # ==========================================
    print("-" * 50)
    print("Converting JSONL to final Parquet for checking...")
    
    # 讀取完整的 JSONL
    try:
        final_data = []
        with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    final_data.append(json.loads(line))
                except:
                    continue
        
        df_final = pd.DataFrame(final_data)
        final_parquet_name = "ollama_section_labels_final.parquet"
        df_final.to_parquet(final_parquet_name)
        print(f"✅ Final merged file saved to: {final_parquet_name} ({len(df_final)} rows)")
        
        # 檢查結構
        if len(df_final) > 0:
            sample = df_final.iloc[0]
            print(f"Sample Article: {sample['article_id']}")
            # print(sample['section_labels']) # 內容可能很長，斟酌列印
            print("Structure check passed.")
            
    except Exception as e:
        print(f"⚠️ Error converting to parquet: {e}")
        print(f"Don't worry, your data is safe in {OUTPUT_JSONL}")