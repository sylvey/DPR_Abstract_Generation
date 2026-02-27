import os, pickle, hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk

CSV_PATH = "for_section_normalization.pkl"
CACHE_DIR = "../data/cache"



def load():
    df = pd.read_pickle(CSV_PATH)
    df = df.dropna(subset=["sections"])

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    pubmed_train = Dataset.from_pandas(train_df[["sections"]].reset_index(drop=True))
    pubmed_val   = Dataset.from_pandas(val_df[["sections"]].reset_index(drop=True))
    pubmed_test  = Dataset.from_pandas(test_df[["sections"]].reset_index(drop=True))
    return pubmed_train, pubmed_val, pubmed_test


LLAMA3_CHAT_TEMPLATE = r"""{% if messages[0]['role'] != 'system' %}
{% set messages = [{'role':'system','content':((
                    "You are a biomedical paragraph annotator trained to generate pargraph labels based on the provided sections from scientific articles. "
                    "Please assign each paragraph a label that best describes its content. "
                    "Below are the labels you can choose from:\n"
                    "'background', 'objective' \n"
                    "'study_samples', 'procedure', 'Prior_work_entity', 'statisical_method' \n"
                    "'results_findings', 'data_statistics', 'novel_entity' \n"
                    "'interpretation', 'implication', 'limitation', 'recommendation','future_work','conclusion' \n"
                    "labels should be providesed in a list format. "
                    "example: "
                    "Paragraphs: ['Paragraph 1 text...', 'Paragraph 2 text...', ....] "
                    "Labels: ['study_samples', 'results_findings', ....] "
                    ""

                ))}] + messages%}
{% endif %}
{% for m in messages %}
{{ '<|start_header_id|>' + m['role'] + '<|end_header_id|>\n\n' + m['content'] + '<|eot_id|>' }}
{% endfor %}
{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""

def ensure_llama3_template(tokenizer):
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    return tokenizer



