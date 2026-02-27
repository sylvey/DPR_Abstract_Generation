import json
import re
import argparse
from disco_score import DiscoScorer

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="../gpt/outputs/pairs.jsonl",
                        help="Path to the input file (json or jsonl)")
    parser.add_argument("--ref", default="ref", help="Input JSONL file")
    parser.add_argument("--pred", default="pred", help="Input JSONL file")
    args = parser.parse_args()

    device = "cuda:0"
    model_name = "bert-base-uncased" 
    
    print(f"Initializing DiscoScorer with {model_name}...")
    scorer = DiscoScorer(device=device, model_name=model_name)
    
    input_file = args.pairs
    system_preds = []
    references = []

    print(f"Loading data from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0) 

        if first_char == '[':
            
            print("Detected JSON list format.")
            data = json.load(f)
        else:
            print("Detected JSONL line-by-line format.")
            data = [json.loads(line) for line in f if line.strip()]

        for entry in data:
            gen_text = entry.get(args.pred, "")
            ref_text = entry.get(args.ref, "") 
            
            if not gen_text or len(gen_text) < 10:
                continue
                
            clean_gen = clean_text(gen_text)
            clean_ref = clean_text(ref_text)
            
            system_preds.append(clean_gen)
            references.append([clean_ref])

    print(f"Evaluating {len(system_preds)} samples...")

    if len(system_preds) > 0:
        print(f"Calculating DiscoScore (Sent & Focus) for {len(system_preds)} samples...")
        
        sent_scores = []
        focus_scores = []

        for i, (sys, ref) in enumerate(zip(system_preds, references)):
            try:
                
                s_score = scorer.DS_SENT_NN(sys, ref)
                sent_scores.append(s_score)
                
                f_score = scorer.DS_Focus_NN(sys, ref)
                focus_scores.append(f_score)
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(system_preds)} samples...")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        print("-" * 30)
        print(f"Results for {input_file}:")
        
        if sent_scores:
            avg_sent = sum(sent_scores) / len(sent_scores)
            print(f"Average DS_SENT_NN (Coherence Flow):   {avg_sent:.4f}")
        else:
            print("No valid DS_SENT_NN scores.")

        if focus_scores:
            avg_focus = sum(focus_scores) / len(focus_scores)
            print(f"Average DS_Focus_NN (Entity Content): {avg_focus:.4f}")
        else:
            print("No valid DS_Focus_NN scores.")
            
        print(f"Total processed: {len(sent_scores)}")
        print("-" * 30)
            
    else:
        print("No valid data found in input file.")

if __name__ == "__main__":
    main()