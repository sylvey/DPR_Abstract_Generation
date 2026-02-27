import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import PeftModel

from evaluate import load as load_metric
from data import load
from utilities import batched_generate, build_infer_prompts
from trl import SFTTrainer, SFTConfig


def main():
    
    MAX_SEQ_LENGTH = 2048
    
    ## ----------model-----------------##

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "../llama/models/unsloth-llama-3.2-3b",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    # model = PeftModel.from_pretrained(base, "models/unsloth-llama-3.2-3b")  # 掛上 adapter
    model.eval()
   
    ## --------------------------------##

    ## --------data preparation -------##
    _, _, test_set = load()
    test_set = test_set.select(range(80))
    prompts = build_infer_prompts(tokenizer, test_set, max_seq_length=MAX_SEQ_LENGTH)

    preds = batched_generate(
        model, tokenizer, prompts,
        max_seq_length=MAX_SEQ_LENGTH,
        max_new_tokens=512, num_beams=1, batch_size=2
    )
    refs = [x["sections"] for x in test_set] 

    ## --------------------------------##
    
    # ---- Save predictions & references ----
    import json, os

    os.makedirs("outputs", exist_ok=True)
    preds_path = "outputs/preds.txt"
    refs_path  = "outputs/refs.txt"
    jsonl_path = "outputs/pairs.jsonl"

    # refs 你前面已經有: refs = [x["abstract"] for x in test_set]
    # 若資料集中有 id，就一併存；沒有就用索引
    def _get_id(x, i):
        try:
            return x.get("id", i)
        except Exception:
            return i

    with open(preds_path, "w", encoding="utf-8") as f_pred, \
         open(refs_path,  "w", encoding="utf-8") as f_ref, \
         open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for i, (p, r) in enumerate(zip(preds, refs)):
            # 保險起見去掉換行
            p = str(p).rstrip("\n")
            r = str(r).rstrip("\n")
            f_pred.write(p + "\n")
            f_ref.write(r + "\n")
            rec = {
                "id": _get_id(test_set[i], i),
                "pred": p,
                "ref": r,
            }
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Saved] {preds_path}, {refs_path}, {jsonl_path}")





    # ## --------test--------------------##
    rouge = load_metric("rouge")
    refs = [x["abstract"] for x in test_set]
    rouge_res = rouge.compute(predictions=preds, references=refs,
                              rouge_types=["rouge1","rouge2","rougeL"], use_aggregator=True)
    print("ROUGE-1:", rouge_res["rouge1"])
    print("ROUGE-2:", rouge_res["rouge2"])
    print("ROUGE-L:", rouge_res["rougeL"])

    # # -------- BERTScore (SciBERT) --------#
    # # Make sure ../led/models/scibert is a valid HF model dir
    # # (has config.json, pytorch_model.bin, vocab.txt, etc.)
    # bertscore = load_metric("bertscore")

    # # BERTScore works on raw strings; keep your preds/refs as-is.
    # # SciBERT is English-domain, so set lang='en'.
    # # You can tune batch_size if you hit GPU RAM limits.
    # bs_res = bertscore.compute(
    #     predictions=preds,
    #     references=refs,
    #     model_type="../led/models/scibert",  # <- your local SciBERT path
    #     lang="en",
    #     idf=True,              # IDF weighting often helps for summarization
    #     batch_size=8,          # adjust if OOM; 8 is a reasonable start
    #     num_layers=9,    
    #     device="cuda:0" if torch.cuda.is_available() else "cpu"
    # )

    # # bs_res has lists of per-example precision/recall/f1 — take means
    # import numpy as np
    # print("BERTScore-P:", float(np.mean(bs_res["precision"])))
    # print("BERTScore-R:", float(np.mean(bs_res["recall"])))
    # print("BERTScore-F1:", float(np.mean(bs_res["f1"])))


    ## --------------------------------##

    return 


if __name__ == "__main__":
    main()