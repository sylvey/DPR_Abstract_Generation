import argparse
import json
import numpy as np
import pickle
import torch
from tqdm import tqdm

from alignscore import AlignScore

def evaluate_factuality_batch(summaries, sources, device="cuda" if torch.cuda.is_available() else "cpu"):
    
    align_model = AlignScore(model='roberta-base', batch_size=16, device=device, ckpt_path="AlignScore/AlignScore-base.ckpt")

    print(f"Starting evaluation on {len(summaries)} pairs using {device}...")


    print("Calculating AlignScore...")
    align_scores = align_model.score(sources, summaries)

    results = {
        "avg_alignscore": np.mean(align_scores),
        "raw_alignscore": align_scores
    }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="outputs/pairs.jsonl", help="Input JSONL file containing predictions")
    parser.add_argument("--id_type", default="article_id", help="ID field to match with source text")
    parser.add_argument("--pred", default="pred", help="Field name for generated summary")
    
    args = parser.parse_args()

    if args.id_type in ['abstract', 'reference']:
        with open("abstract_mappings.pkl", 'rb') as f:
            id_dict = pickle.load(f)
    else:
        with open("mappings.pkl", 'rb') as f:
            loaded_index_dict, loaded_article_dict = pickle.load(f)
            id_dict = loaded_index_dict if args.id_type == 'id' else loaded_article_dict

    preds = []
    refs = []

    with open(args.pairs, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            p = str(record[args.pred]).replace("\n", " ").strip()
            r = str(id_dict[record[args.id_type]]).replace("\n", " ").strip()
            
            if p and r: 
                preds.append(p)
                refs.append(r)

    results = evaluate_factuality_batch(preds, refs)

    print(f"\n[Evaluation Finished] {len(preds)} pairs evaluated.")
    print(f"Average AlignScore: {results['avg_alignscore']:.4f}")


if __name__ == "__main__":
    main()