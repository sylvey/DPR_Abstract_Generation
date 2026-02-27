import os, pickle, hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk

CSV_PATH = "../for_section_normalization.pkl"
CACHE_DIR = "../data/cache"



def load():
    

    df = pd.read_pickle(CSV_PATH)
    
    df = df.dropna(subset=["sections"])
    
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    sections_series = test_df["sections"]
    flat_sections = sections_series.explode().explode() 
    flat_df = flat_sections.to_frame(name="sections").reset_index(drop=True)
    pubmed_test = Dataset.from_pandas(flat_df)

    pubmed_train = Dataset.from_pandas(train_df[["sections"]].reset_index(drop=True))
    pubmed_val   = Dataset.from_pandas(val_df[["sections"]].reset_index(drop=True))
    # pubmed_test  = Dataset.from_pandas(test_df[["sections"]].reset_index(drop=True))
    
    return pubmed_train, pubmed_val, pubmed_test






