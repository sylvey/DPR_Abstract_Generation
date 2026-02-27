import os, pickle, hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk

PARQUEST_PATH = "dataset/all_articles5-v2_sections_combined.parquet"


def load(file_path=PARQUEST_PATH, split = True):
    
    if file_path.endswith(".pkl"):
        df = pd.read_pickle(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".jsonl"):
        df = pd.read_json(file_path, lines=True)
    
    keep_columns = ["article_id", "abstract", "sections_combined"]

    if not split:
        dataset = Dataset.from_pandas(df[keep_columns].reset_index(drop=True))
        return dataset
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    

    pubmed_train = Dataset.from_pandas(train_df[keep_columns].reset_index(drop=True))
    pubmed_val   = Dataset.from_pandas(val_df[keep_columns].reset_index(drop=True))
    pubmed_test  = Dataset.from_pandas(test_df[keep_columns].reset_index(drop=True))
    
    return pubmed_train, pubmed_val, pubmed_test

