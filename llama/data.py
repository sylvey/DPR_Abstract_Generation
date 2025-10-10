import os, pickle, hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk

CSV_PATH = "../data/all_articles5-v2.csv"
CACHE_DIR = "../data/cache"



def load():
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={"full_text": "article", "abstract": "abstract"})
    df = df.dropna(subset=["article", "abstract"])
    df = df[df["article"].str.strip().astype(bool)]
    df = df[df["abstract"].str.strip().astype(bool)]
    df["article"] = df["article"].astype(str)
    df["abstract"] = df["abstract"].astype(str)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    pubmed_train = Dataset.from_pandas(train_df[["article", "abstract"]].reset_index(drop=True))
    pubmed_val   = Dataset.from_pandas(val_df[["article", "abstract"]].reset_index(drop=True))
    pubmed_test  = Dataset.from_pandas(test_df[["article", "abstract"]].reset_index(drop=True))
    return pubmed_train, pubmed_val, pubmed_test

def generate_conversation(examples):
    problems  = examples["article"]
    solutions = examples["abstract"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role": "user", "content": "generate the abstract of this:" + problem},
            {"role": "assistant", "content": solution},
        ])
    return {"conversations": conversations}


LLAMA3_CHAT_TEMPLATE = r"""{% if messages[0]['role'] != 'system' %}
{% set messages = [{'role':'system','content':'You are a helpful assistant for biomedical article abstract generation.'}] + messages %}
{% endif %}
{% for m in messages %}
{{ '<|start_header_id|>' + m['role'] + '<|end_header_id|>\n\n' + m['content'] + '<|eot_id|>' }}
{% endfor %}
{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""

def ensure_llama3_template(tokenizer):
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    return tokenizer

def _fingerprint(tokenizer):
    tok_id = getattr(tokenizer, "name_or_path", "unknown")
    chat_tpl = getattr(tokenizer, "chat_template", "")
    try:
        stat = os.stat(CSV_PATH)
        data_sig = f"{stat.st_mtime}-{stat.st_size}"
    except FileNotFoundError:
        data_sig = "no_csv"
    raw = f"{tok_id}|{hashlib.md5(str(chat_tpl).encode()).hexdigest()}|{data_sig}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]

def data_prep(tokenizer, force_rebuild=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = os.path.join(CACHE_DIR, f"conv_cache_{_fingerprint(tokenizer)}")
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "val")
    test_dir  = os.path.join(base, "test")

    if (not force_rebuild) and all(os.path.exists(p) for p in [train_dir, val_dir, test_dir]):
        print(f"✅ Loaded conversations from cache: {base}")
        return load_from_disk(train_dir), load_from_disk(val_dir), load_from_disk(test_dir)

    # 重新建立
    ensure_llama3_template(tokenizer)

    train_set, val_set, test_set = load()

    def to_conv(ds):
        conv = ds.map(generate_conversation, batched=True)["conversations"]
        texts = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        return Dataset.from_dict({"text": list(texts)})


    train_dataset = to_conv(train_set)
    val_dataset   = to_conv(val_set)
    test_dataset  = to_conv(test_set)

    os.makedirs(base, exist_ok=True)
    train_dataset.save_to_disk(train_dir)
    val_dataset.save_to_disk(val_dir)
    test_dataset.save_to_disk(test_dir)
    print(f"💾 Saved conversations to: {base}")

    return train_dataset, val_dataset, test_dataset
    
