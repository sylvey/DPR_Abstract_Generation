import json
import spacy
import argparse
import sys
from scispacy.linking import EntityLinker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="outputs/pairs.jsonl", help="Input JSONL file")
    parser.add_argument("--ref", default="ref", help="Input JSONL file")
    parser.add_argument("--pred", default="pred", help="Input JSONL file")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top UMLS candidates to consider")
    args = parser.parse_args()

    print("Loading scispacy model (en_core_sci_md)...")
    try:
        nlp = spacy.load("en_core_sci_md")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

    linker = nlp.get_pipe("scispacy_linker")

    def get_umls_cuis(text, top_k=1):
        doc = nlp(text)
        cuis = set()
        for ent in doc.ents:
            if ent._.kb_ents:
                candidates = [cui for cui, score in ent._.kb_ents[:top_k]]
                cuis.update(candidates)
        return cuis

    input_file = args.pairs
    print(f"Reading data from: {input_file}")
    
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        return

    print(f"Total entries loaded: {len(data)}")
    
    if len(data) == 0:
        print("File is empty!")
        return

    print("\n" + "="*40)
    print("DIAGNOSING FIRST ENTRY")
    print("="*40)
    
    first_entry = data[0]
    print(f"Keys in JSON: {list(first_entry.keys())}")
    
    gen_text = first_entry.get(args.pred) or ""
    ref_text = first_entry.get(args.ref) or ""
    
    print(f"Gen Text Length: {len(gen_text)}")
    print(f"Ref Text Length: {len(ref_text)}")
    
    if len(ref_text) > 0:
        print(f"Ref Text Preview: {ref_text[:100]}...")
        
        doc = nlp(ref_text)
        print(f"\nRaw Entities Found (Before Filter): {len(doc.ents)}")
        print(f"First 5 Raw Entities: {[e.text for e in doc.ents[:5]]}")
        
        clean_ents = set()
        for e in doc.ents:
            lemma = e.lemma_.lower()
            if len(lemma) < 3: continue
            clean_ents.add(lemma)
            
        print(f"Clean Entities (After Filter): {len(clean_ents)}")
        print(f"First 5 Clean Entities: {list(clean_ents)[:5]}")
    else:
        print("WARNING: Reference text is empty! Check JSON keys.")

    print("="*40 + "\n")

    valid_samples = 0
    total_recall = 0
    
    for i, entry in enumerate(data): 
        ref_text = entry.get(args.ref, "")
        gen_text = entry.get(args.pred, "")
        
        if not ref_text: continue
        
        doc_ref = nlp(ref_text)
        doc_gen = nlp(gen_text)
        
        ents_ref = get_umls_cuis(ref_text, top_k=args.top_k)
        ents_gen = get_umls_cuis(gen_text, top_k=args.top_k)
        
        if len(ents_ref) > 0:
            intersection = ents_ref.intersection(ents_gen)
            recall = len(intersection) / len(ents_ref)
            total_recall += recall
            valid_samples += 1
            
    print(f"Test Run Valid Samples: {valid_samples}")
    if valid_samples > 0:
        print(f"Test Run Avg Recall: {total_recall / valid_samples:.4f}")

if __name__ == "__main__":
    main()