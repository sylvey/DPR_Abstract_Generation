from unsloth import FastLanguageModel
import torch
from data import load, ensure_llama3_template  # 直接用你 data.py 裡的
# from torch.nn.attention import sdpa_kernel
from contextlib import nullcontext


def _sdpa_ctx():
    """
    回傳一個可用的 context manager：
    - torch>=2.4: 使用 torch.nn.attention.sdpa_kernel(SDPBackend.MATH)
    - 舊版: 使用 torch.backends.cuda.sdp_kernel(enable_*)
    - 都不可用時: 回傳 nullcontext()
    """
    try:
        # 新 API（有些版本回傳的不是 context manager，要檢查）
        from torch.nn.attention import sdpa_kernel, SDPBackend
        ctx = sdpa_kernel(SDPBackend.MATH)
        return ctx if hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__") else nullcontext()
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel as sdpa_kernel_old
            ctx = sdpa_kernel_old(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            return ctx if hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__") else nullcontext()
        except Exception:
            return nullcontext()


def generate_text(model, text, tokenizer):
    inputs = tokenizer(text, return_tensors = "pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens = 20)
    print(tokenizer.decode(outputs[0], skip_special_tokens = True))


def build_infer_prompts(tokenizer, test_set, max_seq_length=2048):
    """只做「user 提問」的 prompt；不包含答案。"""
    ensure_llama3_template(tokenizer)

    msgs = []
    for ex in test_set:
        msgs.append([
            {"role": "user", "content": f"Paragraphs:{ex['sections']} \nLabels:"}
        ])
    # 用 chat template 串成文字，並加上 generation prompt
    prompts = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return list(prompts)

def batched_generate(model, tokenizer, prompts, max_seq_length=2048,
                     max_new_tokens=512, num_beams=1, batch_size=2):

    FastLanguageModel.for_inference(model)
    model.generation_config.cache_implementation = "static"  # 防止 beam 下的 KV cache 問題

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i+batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_seq_length).to(model.device)

        with _sdpa_ctx():   # ← 用 context manager 套用 MATH 後端
            try:
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,          # 先設 1；等環境 OK 再開 >1
                        do_sample=False,
                        no_repeat_ngram_size=3,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            except AttributeError as e:
                # 若仍遇到 past_key_values/reorder_cache 之類問題，自動回退 greedy
                if "reorder_cache" in str(e):
                    with torch.no_grad():
                        gen = model.generate(
                            **enc,
                            max_new_tokens=max_new_tokens,
                            num_beams=1,
                            do_sample=False,
                            no_repeat_ngram_size=3,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                else:
                    raise

        start = enc["input_ids"].size(1)
        outs.extend([
            tokenizer.decode(gen[j, start:], skip_special_tokens=True).strip()
            for j in range(gen.size(0))
        ])
    return outs
