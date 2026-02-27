from unsloth import FastLanguageModel
import torch
# from torch.nn.attention import sdpa_kernel
from contextlib import nullcontext
from torch.utils.data import DataLoader
import functools
import re

def create_single_prompt(tokenizer, example):
        msgs = [
            {
                'role': 'system',
                'content': (
                    "You are a biomedical paragraph annotator trained to generate pargraph labels based on the provided sections from scientific articles. "
                    "Please assign each paragraph a label that best describes its content. "
                    "Below are the labels you can choose from:\n"
                    "'background', 'objective' \n"
                    "'study_samples', 'procedure', 'Prior_work_entity', 'statisical_method' \n"
                    "'results_findings', 'data_statistics', 'novel_entity' \n"
                    "'interpretation', 'implication', 'limitation', 'recommendation','future_work','conclusion' \n"
                    "labels should be in a list format like. \n"
                    "example: "
                    "Paragraph: 'In this study, we investigated...  ' \n"
                    "Labels: ['conclusion', 'interpretation', ....] "
                )
            },
            {
                'role': 'user',
                'content': (
                    f"Paragraph:{example['sections']} \n"
                    "Labels:"
                )
            }

        ]
        # 注意：我們使用 apply_chat_template 僅建立一個序列
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, reasoning_effort="low" )
        new_ending = "<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
        assert prompt.rstrip().endswith(new_ending), \
            f"Chat template 結尾錯誤！\n預期: ...{new_ending}\n實際: ...{prompt.rstrip()[-len(new_ending)-10:]}"

        return {"input_text": prompt}

def collate_fn(model, tokenizer, max_seq_length, batch):
    # 提取每個樣本的 prompt 文本
    texts = [item["input_text"] for item in batch]
    
    # 對整個批次的文本進行 tokenize 和 padding
    tokenized = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_seq_length # 使用傳入的 max_seq_length
    )
    # 將 tokenized results (包含 input_ids, attention_mask) 移到 model.device
    return {k: v.to(model.device) for k, v in tokenized.items()}




_PLANNING_TOKEN_PATTERN = re.compile(r"^(analysis|assistant\w*)[:\s]", re.IGNORECASE)


import re # 確保 re 已經在檔案頂部 import

def _strip_planning_prefix(text: str) -> str:
    s_cleaned = text.lstrip()
    if not s_cleaned: 
        return s_cleaned

    # 擴展你的禁用詞列表，包含範例中出現的詞
    b_list_sent = (
        # 來自你 banned_anywhere 的詞
        "analysis", "assistant", "assistantanalysis", "assistantfinal",
        "we need", "we should", "let's", "let us", "we must",
        "must ensure", "ensure no", "task:", "goal:", "strategy", "plan",
        "this article", "the author", "the authors", "in this editorial",
        "in this letter", "in this commentary", "in this review",
        
        # 來自你 Preamble 範例的特定片語
        "we need to produce", "we need to summarise", "we need to summarize", 
        "it's a review", "it is a review", "it's a summary", "it is a summary",
        "summarize the topic", "capture the central argument", 
    )

    while True: 
        # 尋找第一個句子/區塊的結尾 (句點, 問號, 驚嘆號, 或 換行符)
        m = re.search(r"[\.\?\!\n]", s_cleaned)
        
        if not m:
            # 已經沒有終止符了，檢查剩下的整個字串
            if any(b in s_cleaned.lower() for b in b_list_sent):
                return "" # 整個字串都是 Preamble
            else:
                break 

        first_segment_text = s_cleaned[:m.end()]
        first_segment_lower = first_segment_text.lower()
        if any(b in first_segment_lower for b in b_list_sent):
            s_cleaned = s_cleaned[m.end():].lstrip()
            continue
        if len(re.findall(r"\w+", first_segment_text)) < 2:
            s_cleaned = s_cleaned[m.end():].lstrip()
            continue
        else:
            break
    
    return s_cleaned.lstrip('"\'' + " ") 

# Add these imports to the top of your file
import os
import json
import functools
from torch.utils.data import DataLoader
from tqdm import tqdm

# (Your existing create_single_prompt and collate_fn functions remain the same)
# ...

def batched_generate(test_set, model, tokenizer):
    """
    Generates predictions, saves them with references to a file as it goes,
    and supports automatic resumption.
    """
    output_dir = "outputs"
    out_path = os.path.join(output_dir, "pairs.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Resumption Logic ----
    start_index = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            start_index = sum(1 for _ in f) # Count existing lines
    
    if start_index >= len(test_set):
        print(f"✅ Generation already complete. Found {start_index} entries in {out_path}.")
        return

    if start_index > 0:
        print(f"✅ Resuming generation from index {start_index}...")
    else:
        print(f"🚀 Starting new generation...")

    # ---- Data Preparation for Remaining Items ----
    
    # Only process the part of the dataset that is not yet completed
    # test_set_to_process = test_set.select(range(start_index, len(test_set)))
    test_set_to_process = test_set.select(range(start_index, 80))

    partial_create_prompt = functools.partial(create_single_prompt, tokenizer)
    # Important: Do NOT remove the 'abstract' column here, we need it for the reference!
    test_set_prompts = test_set_to_process.map(partial_create_prompt, load_from_cache_file=False)
    
    BATCH_SIZE = 1 # Your original batch size
    MAX_SEQ_LENGTH = 2048

    partial_collate_fn = functools.partial(
        collate_fn, 
        model, 
        tokenizer, 
        MAX_SEQ_LENGTH
    )

    dataloader = DataLoader(
        test_set_prompts, 
        batch_size=BATCH_SIZE, 
        collate_fn=partial_collate_fn,
        shuffle=False # Ensure order is preserved
    )
    
    # We need the references from the unprocessed part of the dataset
    refs_to_process = test_set_to_process["sections"]

    # ---- Generation and Immediate Saving Loop ----
    try:
        # Open the file in append mode to add new entries
        with open(out_path, "a", encoding="utf-8") as f:
            # Enumerate to keep track of the index for the reference
            
            for i, batch in enumerate(tqdm(dataloader, desc="Generating and Saving")):
                with torch.no_grad():
                    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
                    gen_out_batch = model.generate(
                        **batch,
                        max_new_tokens=500,
                        temperature=0.7,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=end_id if end_id is not None and end_id != tokenizer.unk_token_id else None,
                    )
                
                # This loop handles decoding for each item in the batch (even if BATCH_SIZE > 1)
                for j in range(gen_out_batch.sequences.size(0)):
                    input_ids = batch["input_ids"][j]
                    
                    # Decode the generated part only
                    prompt_len = int((input_ids != tokenizer.pad_token_id).sum().item())
                    gen_ids = gen_out_batch.sequences[j, prompt_len:]
                    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                    # Remove unwanted planning chatter sometimes emitted before the actual abstract.
                    pred = _strip_planning_prefix(pred)
                    
                    # Get the corresponding reference using the global index
                    # i = batch index, j = index within batch
                    ref_index = i * BATCH_SIZE + j
                    ref = str(refs_to_process[ref_index]).strip()

                    # Write the pair to the file immediately
                    f.write(json.dumps({"pred": pred, "ref": ref}, ensure_ascii=False) + "\n")
                    f.flush()

    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user. Progress has been saved.")
    
    print(f"\n✨ Generation process finished. Output is in {out_path}")
    return
