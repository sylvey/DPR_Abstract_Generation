import argparse
import json
import numpy as np
import pickle
import torch
from tqdm import tqdm

# 匯入事實性評估庫
from summac.model_summac import SummaCConv


def evaluate_factuality_batch(summaries, sources, device="cuda" if torch.cuda.is_available() else "cpu"):
   
    summac_model = SummaCConv(models=["vitc"], bin_number=10, granularity="sentence", device=device)
    
    print("Calculating SummaC...")
    summac_results = summac_model.score(sources, summaries)
    summac_scores = summac_results["scores"]

    results = {
        "avg_summac": np.mean(summac_scores),
        "raw_summac": summac_scores,
    }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="outputs/pairs.jsonl", help="Input JSONL file containing predictions")
    parser.add_argument("--id_type", default="article_id", help="ID field to match with source text")
    parser.add_argument("--pred", default="pred", help="Field name for generated summary")
    # parser.add_argument("--output", default="factuality_results.json", help="Path to save results")
    
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
    print(f"Average SummaC Score: {results['avg_summac']:.4f}")

    with open(args.pairs.replace(".jsonl", "_factuality_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Detailed scores saved to {args.pairs.replace('.jsonl', '_factuality_results.json')}")

if __name__ == "__main__":
    main()