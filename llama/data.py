import os, pickle, hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk

CSV_PATH = "../data/all_articles5-v2-publication_type.csv"
CACHE_DIR = "../data/cache"



def load():
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={"full_text": "article", "positive_labels": "publication_type"})
    df = df.dropna(subset=["article", "abstract", "publication_type"])
    df = df[df["article"].str.strip().astype(bool)]
    df = df[df["abstract"].str.strip().astype(bool)]
    df["article"] = df["article"].astype(str)
    df["abstract"] = df["abstract"].astype(str)
    df['publication_type'] = df['publication_type'].astype(str)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    pubmed_train = Dataset.from_pandas(train_df[["article", "abstract", "publication_type"]].reset_index(drop=True))
    pubmed_val   = Dataset.from_pandas(val_df[["article", "abstract", "publication_type"]].reset_index(drop=True))
    pubmed_test  = Dataset.from_pandas(test_df[["article", "abstract", "publication_type"]].reset_index(drop=True))
    return pubmed_train, pubmed_val, pubmed_test

def generate_conversation(examples):
    problems  = examples["article"]
    solutions = examples["abstract"]
    publication_types = examples["publication_type"]

    conversations = []
    for problem, solution, publication_type in zip(problems, solutions, publication_types):
        conversations.append([
            {"role": "user", "content": f"publication_type: {publication_type} \narticle: {problem} \n\nJSON Output:"},
            {"role": "assistant", "content": "{"+f"abstract: {solution}" +"}"},
        ])
    return {"conversations": conversations}


LLAMA3_CHAT_TEMPLATE = r"""{% if messages[0]['role'] != 'system' %}
{% set messages = [{'role':'system','content':((
                    "You are a biomedical summarization assistant trained to generate accurate, publication-quality abstracts. "
                    "Your goal is to produce a clear, concise summary in scientific language that accurately reflects the source text. "
                    "Do not invent or infer information not explicitly stated.\n\n"
                    "Adjust your tone, structure, and level of detail according to the publication type:\n"
                    "- **Primary research articles:** Summarize the background, objectives, study design, key methods, main results, and conclusions.\n"
                    "- **Case reports:** Describe the patient(s), clinical presentation, diagnostic approach, intervention, and outcome.\n"
                    "- **Reviews (systematic or narrative):** Summarize the topic focus, scope of the literature reviewed, main findings or themes, and key conclusions or implications.\n"
                    "- **Letters, commentaries, or editorials:** Capture the central argument, critique, or viewpoint being expressed, and briefly note any evidence or context discussed.\n"
                    "- **Published erratums or corrections:** Identify the correction being made and, if available, its impact on the original findings or interpretation.\n\n"
                    "Formatting and style requirements:\n"
                    "- Write one coherent paragraph unless otherwise noted.\n"
                    "- For **primary research articles** or **reviews**, write about 150–300 words to capture the main points thoroughly.\n"
                    "- For **case reports**, **letters**, **commentaries**, **editorials**, or **erratums**, write about 100–200 words.\n"
                    "- You MUST respond with a single, valid JSON object.\n"
                    "- This JSON object must contain one key: \"abstract\".\n"
                    "- The value of the \"abstract\" key must be the generated summary string.\n"
                    "- Example of a perfect response for a primary research article:\n"
                    "{\"abstract\": \"This study evaluated the efficacy of a new drug.... We conducted a randomized controlled trial with 200 patients... The drug showed a 50% improvement over placebo (p < 0.05)... This new drug is a promising treatment.\"}\n\n"
                    ))}] + messages %}
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
    
