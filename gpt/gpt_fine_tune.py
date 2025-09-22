import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import os
import pickle
#!/usr/bin/env python3
from datasets import load_dataset
import torch
from contextlib import nullcontext
from evaluate import load

import shutil
import transformers
from transformers import AutoModelForCausalLM

import numpy as np

from transformers import DataCollatorWithPadding
from contextlib import nullcontext

from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
from peft import LoraConfig, get_peft_model, PeftModel
import math
import random
from datasets import Dataset, DatasetDict

import os
import json
import torch
from safetensors.torch import save_file
from peft import get_peft_model_state_dict
import numbers

def _to_jsonable(x):
    # 遞迴把物件轉成 JSON 能接受的東西
    if x is None or isinstance(x, (bool, str, numbers.Number)):
        return x
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, set):
        return [_to_jsonable(v) for v in sorted(list(x), key=lambda v: str(v))]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, torch.dtype):
        return str(x)
    # 其他不認得的型別 → 字串（保底）
    return str(x)

def save_lora_adapters_skip_meta(model, out_dir: str, adapter_name: str = "default"):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 取 LoRA 權重，跳過 meta tensor
    state = get_peft_model_state_dict(model, adapter_name=adapter_name)
    filtered, skipped = {}, []
    for k, v in state.items():
        if getattr(v, "is_meta", False):
            skipped.append(k)
            continue
        filtered[k] = v.detach().to("cpu")
    save_file(filtered, os.path.join(out_dir, "adapter_model.safetensors"))

    # 2) 存 adapter config（先做 JSON 化）
    cfg = model.peft_config.get(adapter_name, None)
    if cfg is not None:
        cfg_dict = _to_jsonable(cfg.to_dict())
        with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)

    print(f"✅ Saved LoRA adapter to: {out_dir} (skipped {len(skipped)} meta tensors)")
    if skipped:
        print("   ↳ skipped keys (meta):", skipped[:5], "..." if len(skipped) > 5 else "")

import torch




def main():
    # 載入資料
    global tokenizer, led 
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


    print("transformers version:", transformers.__version__)
    print("transformers file  :", transformers.__file__)

    # torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = True

    cache_path = "../data/pubmed_datasets_5000_500_500.pkl"

    if os.path.exists(cache_path):
        print(f"✅ Loading datasets from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            pubmed_train, pubmed_val, pubmed_test = pickle.load(f)

    else:
        print("🚀 No cache found. Processing from CSV...")
        df = pd.read_csv('../data/all_articles5-v2.csv')

        # rename + clean
        df = df.rename(columns={
            "full_text": "article",
            "abstract": "abstract"
        })
        df = df.dropna(subset=["article", "abstract"])
        df = df[df["article"].str.strip().astype(bool)]
        df = df[df["abstract"].str.strip().astype(bool)]
        df["article"] = df["article"].astype(str)
        df["abstract"] = df["abstract"].astype(str)

        # split
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        pubmed_train = Dataset.from_pandas(train_df[["article", "abstract"]].reset_index(drop=True))
        pubmed_val   = Dataset.from_pandas(val_df[["article", "abstract"]].reset_index(drop=True))
        pubmed_test  = Dataset.from_pandas(test_df[["article", "abstract"]].reset_index(drop=True))

        pubmed_train = pubmed_train.select(range(min(len(pubmed_train), 8000)))
        pubmed_val   = pubmed_val.select(range(min(len(pubmed_val), 800)))
        pubmed_test  = pubmed_test.select(range(min(len(pubmed_test), 800)))

        tokenizer = AutoTokenizer.from_pretrained("./led_pubmed_model")
        encoder_max_length = 512
        decoder_max_length = 512

        def process_data_to_model_inputs(example):
            inputs = tokenizer(
                example["article"],
                padding="max_length",
                truncation=True,
                max_length=encoder_max_length,
            )
            outputs = tokenizer(
                example["abstract"],
                padding="max_length",
                truncation=True,
                max_length=decoder_max_length,
            )
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            labels = outputs.input_ids

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "global_attention_mask": [1] + [0] * (len(input_ids) - 1),
                "labels": [-100 if token == tokenizer.pad_token_id else token for token in labels]
            }

        pubmed_train = pubmed_train.map(
            process_data_to_model_inputs,
            batched=False,
            remove_columns=["article", "abstract"],
            load_from_cache_file=False,
            desc="🧪 Mapping training data"
        )
        pubmed_val = pubmed_val.map(
            process_data_to_model_inputs,
            batched=False,
            remove_columns=["article", "abstract"],
            load_from_cache_file=False,
            desc="🧪 Mapping val data"
        )

        # 存快取
        with open(cache_path, "wb") as f:
            pickle.dump((pubmed_train, pubmed_val, pubmed_test), f)
        print(f"💾 Saved processed datasets to {cache_path}")

    print("✅ pubmed_train size:", len(pubmed_train))
    print("✅ pubmed_val size:", len(pubmed_val))
    print("✅ pubmed_test size:", len(pubmed_test))
    print(pubmed_train[0])


    # ---- 0) 基本設定 ----
    MODEL_ID = "/jet/home/slin23/tmp_ondemand_ocean_cis230089p_symlink/slin23/full_text_label/gpt/gpt-oss-20b"   # TODO: 換成實際模型ID
    MAX_INPUT_TOKENS = 512
    MAX_TARGET_TOKENS = 64
    MAX_NEW_TOKENS   = 48
    PROMPT_TEMPLATE = (
        "You are a biomedical research assistant.\n"
        "Task: Generate a concise abstract for the following article section.\n"
        "Keep terminology precise; avoid hallucinations; do not fabricate citations.\n"
        "Article:\n{ARTICLE}\n\nWrite the abstract:\n"
    )

    def ensure_article_abstract(ds):
        """若資料集中沒有 article/abstract 欄位，回傳 None 以觸發重建。"""
        f = set(ds.features.keys())
        return ("article" in f and "abstract" in f)

    # ---- 1) 準備 SFT 資料（指令式）----
    # 你的 train/val 已經被 map 成 LED 的欄位了，可能不再有 article/abstract。
    # 若缺少，這裡會重新從 CSV 讀取並用同樣 random_state=42 重建 splits。
    def rebuild_from_csv_if_needed(train_ds, val_ds, test_ds, csv_path="../data/all_articles5-v2.csv"):
        if ensure_article_abstract(train_ds) and ensure_article_abstract(val_ds) and ensure_article_abstract(test_ds):
            return train_ds, val_ds, test_ds  # 直接沿用

        print("⚠️  Detected missing `article/abstract` in train/val; rebuilding splits from CSV for SFT...")
        df_all = pd.read_csv(csv_path)
        df_all = df_all.rename(columns={"full_text": "article", "abstract": "abstract"})
        df_all = df_all.dropna(subset=["article", "abstract"])
        df_all = df_all[df_all["article"].str.strip().astype(bool)]
        df_all = df_all[df_all["abstract"].str.strip().astype(bool)]
        df_all["article"] = df_all["article"].astype(str)
        df_all["abstract"] = df_all["abstract"].astype(str)

        # 與前面一致的隨機切分
        train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=42)
        val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)

        # 尺度跟前面保持一致（最多 8000/800/800）
        train_df = train_df.iloc[:min(len(train_df), 8000)]
        val_df   = val_df.iloc[:min(len(val_df), 800)]
        test_df  = test_df.iloc[:min(len(test_df), 800)]

        return (
            Dataset.from_pandas(train_df[["article","abstract"]].reset_index(drop=True)),
            Dataset.from_pandas(val_df[["article","abstract"]].reset_index(drop=True)),
            Dataset.from_pandas(test_df[["article","abstract"]].reset_index(drop=True)),
        )

    sft_train_raw, sft_val_raw, sft_test_raw = rebuild_from_csv_if_needed(pubmed_train, pubmed_val, pubmed_test)

    


    # ---- 2) Tokenizer 與 4-bit 量化模型 ----
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    tok.padding_side = "left"
    tok.truncation_side = "left"   # 生成時也更穩

    def _has_min_target_tokens(ex):
        # 目標：至少 2 個 token（含我們已經會加上的 eos）
        tgt = tok(ex["abstract"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        return len(tgt) >= 4

    print("🔎 filtering samples with too-short targets (<2 tokens after tokenization)…")
    sft_train_raw = sft_train_raw.filter(_has_min_target_tokens)
    sft_val_raw   = sft_val_raw.filter(_has_min_target_tokens)
    sft_test_raw  = sft_test_raw.filter(_has_min_target_tokens)
    print("sizes after filter:", len(sft_train_raw), len(sft_val_raw), len(sft_test_raw))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # keep compute in bf16
    )

    
    # ... 前略：tok = AutoTokenizer.from_pretrained(MODEL_ID, ...)

    from transformers import AutoModelForCausalLM

    def load_gptoss_mxfaware(model_id: str):
        try:
            # First try: if the model is NOT MXFP4, you can still pass your BnB config here.
            # But because your checkpoint IS MXFP4, this will raise and we’ll fall back.
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,          # ← 改成 fp16
                device_map="auto",
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
        except ValueError as e:
            # MXFP4 + BnB mismatch or similar → reload cleanly WITHOUT BnB
            print("🔁 Detected MXFP4 or quant-config mismatch; reloading without BitsAndBytes…")
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,              # dequantizes MXFP4 to bf16 on V100/Tesla V
                device_map="auto",
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )

    base_model = load_gptoss_mxfaware(MODEL_ID)

   


    # LoRA 設定（可先從保守值跑起）
    lora_cfg = LoraConfig(
        r=8, 
        lora_alpha=4, 
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],  # 若名稱不同，請據實調整
        bias="none", task_type="CAUSAL_LM"
    )
    # base_model.gradient_checkpointing_enable()
    model = get_peft_model(base_model, lora_cfg)

    from torch.nn import Parameter, ParameterList

    def _force_paramlist_to_fp16(mod):
        with torch.no_grad():
            # gate_up_proj: ParameterList of weights [E, dim_in, 2*hidden]
            if hasattr(mod, "gate_up_proj") and isinstance(mod.gate_up_proj, ParameterList):
                for i in range(len(mod.gate_up_proj)):
                    p = mod.gate_up_proj[i]
                    if p.device.type != "meta" and p.dtype != torch.float16:
                        mod.gate_up_proj[i] = Parameter(p.to(torch.float16), requires_grad=p.requires_grad)

            # gate_up_proj_bias: ParameterList of biases [E, 2*hidden]
            if hasattr(mod, "gate_up_proj_bias") and isinstance(mod.gate_up_proj_bias, ParameterList):
                for i in range(len(mod.gate_up_proj_bias)):
                    p = mod.gate_up_proj_bias[i]
                    if p.device.type != "meta" and p.dtype != torch.float16:
                        mod.gate_up_proj_bias[i] = Parameter(p.to(torch.float16), requires_grad=p.requires_grad)

            # gate_down_proj: ParameterList of weights [E, 2*hidden, dim_out]
            if hasattr(mod, "gate_down_proj") and isinstance(mod.gate_down_proj, ParameterList):
                for i in range(len(mod.gate_down_proj)):
                    p = mod.gate_down_proj[i]
                    if p.device.type != "meta" and p.dtype != torch.float16:
                        mod.gate_down_proj[i] = Parameter(p.to(torch.float16), requires_grad=p.requires_grad)

    # 只針對含有 experts 的 MLP 容器執行
    for name, m in model.named_modules():
        if ("experts" in name) or ("moe" in name):
            _force_paramlist_to_fp16(m)


    def _pre_forward_align_to_fp16(mod, inputs):
        # 將 inputs（hidden_states）對齊到 gate_up_proj 的 dtype；同時確保 ParamList 又被轉回 fp16
        _force_paramlist_to_fp16(mod)

        if not inputs:
            return inputs
        x = inputs[0]
        # 代表性參考 dtype：若 gate_up_proj 存在，取第 0 個
        ref_dtype = None
        if hasattr(mod, "gate_up_proj") and isinstance(mod.gate_up_proj, ParameterList) and len(mod.gate_up_proj) > 0:
            ref_dtype = mod.gate_up_proj[0].dtype
        # 預期 ref_dtype 會是 torch.float16；保險對齊一下
        if isinstance(x, torch.Tensor) and (ref_dtype is not None) and x.dtype != ref_dtype:
            x = x.to(ref_dtype)
            # 回傳新的 inputs（PyTorch 2 的 pre_forward hook 支援修改輸入）
            return (x, ) + tuple(inputs[1:])
        return inputs

    # 掛在含有 experts 的模組上
    for name, m in model.named_modules():
        if ("experts" in name) or ("moe" in name):
            try:
                m.register_forward_pre_hook(_pre_forward_align_to_fp16)
            except Exception:
                pass


    left = []
    for n,p in model.named_parameters():
        if (("experts" in n) or ("moe" in n) or ("gate_up_proj" in n) or ("gate_down_proj" in n)) \
        and p.device.type != "meta" and p.is_floating_point() and p.dtype != torch.float16:
            left.append((n, str(p.dtype)))
    print("🎯 leftover non-fp16 in experts:", left[:10])


   

    # 3) 驗證（可保留；只列前幾個殘留 BF16 參數名稱）
    bf16_expert_params = [n for n, p in model.named_parameters()
                        if any(k in n for k in name_hits)
                        and p.is_floating_point() and p.device.type != "meta" and p.dtype == torch.bfloat16]
    if bf16_expert_params:
        print("⚠️ Still BF16 expert params (show up to 8):", bf16_expert_params[:8])

    bf16_expert_buffers = [n for n, b in model.named_buffers()
                        if any(k in n for k in name_hits)
                        and torch.is_floating_point(b) and b.device.type != "meta" and b.dtype == torch.bfloat16]
    if bf16_expert_buffers:
        print("⚠️ Still BF16 expert buffers (show up to 8):", bf16_expert_buffers[:8])


    # 手動把 LoRA B 權重初始化成 0（初始不改變基礎模型）
    with torch.no_grad():
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_B"):
                if hasattr(mod.lora_B, "default") and hasattr(mod.lora_B.default, "weight"):
                    mod.lora_B.default.weight.zero_()
                elif hasattr(mod.lora_B, "weight"):
                    mod.lora_B.weight.zero_()

    def print_float_dtypes(tag, model):
        dtypes = set()
        for n, p in model.named_parameters():
            if p.is_floating_point():
                dtypes.add(p.dtype)
        for n, b in model.named_buffers():
            if torch.is_floating_point(b):
                dtypes.add(b.dtype)
        print(f"[{tag}] floating dtypes in model:", dtypes)
    
    
    # print_float_dtypes("before-cast", model)

    # # 🔧 統一到 float16（Half）
    # for n, p in model.named_parameters():
    #     if p.is_floating_point() and p.dtype != torch.float16:
    #         with torch.no_grad():
    #             p.data = p.data.to(torch.float16)
    # for n, b in model.named_buffers():
    #     if torch.is_floating_point(b) and b.dtype != torch.float16:
    #         with torch.no_grad():
    #             b.data = b.data.to(torch.float16)

    # # 保險：也把 config 標註成 fp16
    # setattr(model.config, "torch_dtype", torch.float16)
    # try:
    #     model = model.to(dtype=torch.float16)
    # except Exception:
    #     pass  # device_map=auto 時，to() 可能不移動參數，但不影響我們已經逐一轉型

    # print_float_dtypes("after-cast", model)

    # 同步 pad token dtype 不影響，但補上這行更一致
    model.config.pad_token_id = tok.pad_token_id

    # --- MoE routing 穩定化（新增這段）---
    for k, v in [("router_top_k", 1), ("router_jitter_noise", 0.0)]:
        if hasattr(model.config, k):
            setattr(model.config, k, v)


    trainable, total = 0, 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"🔧 Trainable params: {trainable:,} / {total:,}")
    assert trainable > 0, "❌ 沒有任何可訓練參數（LoRA target_modules 可能對不到）。"


    def _stabilize_logits(_, __, out):
        # 先把 NaN/Inf 轉成有限數，再夾在 [-50, 50]
        out = torch.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0)
        return out.clamp_(-50, 50)

    # if hasattr(model, "lm_head") and hasattr(model.lm_head, "register_forward_hook"):
    #     model.lm_head.register_forward_hook(_stabilize_logits)

    for m in model.modules():
        # torch.nn.LayerNorm
        if hasattr(m, "eps"):
            m.eps = max(float(m.eps), 1e-5)
        # RMSNorm / 自訂 Norm 常用的欄位名
        if hasattr(m, "variance_epsilon"):
            m.variance_epsilon = max(float(m.variance_epsilon), 1e-5)

    model.config.use_cache = False

    print("Sample attn modules:")
    for n,_ in list(model.named_modules())[:300]:
        if "attn" in n.lower() or "attention" in n.lower():
            print(n)
    # ✅ Re-enable gradient checkpointing to cut activations (works with MoE if non-reentrant)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    except TypeError:
        model.gradient_checkpointing_enable()


    # ---- 3) 指令化樣本映射（並做 label masking：prompt 部分為 -100）----
    def build_prompt(article: str) -> str:
        return PROMPT_TEMPLATE.replace("{ARTICLE}", article)

    def encode_example(article: str, abstract: str):
        prompt = build_prompt(article)
        prompt_ids  = tok(prompt, add_special_tokens=False)["input_ids"]
        target_ids  = tok(abstract + tok.eos_token, add_special_tokens=False)["input_ids"]

        # ---- 安全保底：至少 2 個 target token，否則丟棄該樣本 ----
        if len(target_ids) < 2:
            return {"input_ids": [], "attention_mask": [], "labels": []}  # 讓 map 跳過空樣本


        target_ids  = target_ids[:MAX_TARGET_TOKENS]
        keep_prompt = max(0, MAX_INPUT_TOKENS - len(target_ids))
        prompt_ids  = prompt_ids[-keep_prompt:] if len(prompt_ids) > keep_prompt else prompt_ids

        input_ids = prompt_ids + target_ids
        labels    = ([-100] * len(prompt_ids)) + target_ids
        attn_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }


    def map_sft(ds: Dataset) -> Dataset:
        return ds.map(
            lambda ex: encode_example(ex["article"], ex["abstract"]),
            remove_columns=ds.column_names,
            desc="🧩 Building causal LM samples"
        )
    
    # ==== CACHE: build 指令樣本（map_sft + filter） ====
    def has_label_tokens(ex):
    # 只要 labels 裡有任何一個不是 -100，就保留這個樣本
        return any(t != -100 for t in ex["labels"])
    from datasets import load_from_disk

    # 用模型/Tokenizer & 長度參數組出獨特 cache 路徑，避免撞檔
    tok_id = getattr(tok, "name_or_path", "tok")
    SFT_CACHE_DIR = (
        f"../data/sft_cache_"
        f"{os.path.basename(MODEL_ID)}_"
        f"{os.path.basename(str(tok_id))}_"
        f"in{MAX_INPUT_TOKENS}_tgt{MAX_TARGET_TOKENS}_leftpad"
    )

    if os.path.isdir(SFT_CACHE_DIR):
        print(f"✅ Loading SFT datasets from disk cache: {SFT_CACHE_DIR}")
        sft_train = load_from_disk(os.path.join(SFT_CACHE_DIR, "train"))
        sft_val   = load_from_disk(os.path.join(SFT_CACHE_DIR, "val"))
        sft_test  = load_from_disk(os.path.join(SFT_CACHE_DIR, "test"))
    else:
        print("🧩 Building causal LM samples (first time)...")
        sft_train = map_sft(sft_train_raw).filter(has_label_tokens)
        sft_val   = map_sft(sft_val_raw).filter(has_label_tokens)
        sft_test  = map_sft(sft_test_raw)  # 測試集可不過濾 label

        print(f"💾 Saving SFT datasets to: {SFT_CACHE_DIR}")
        os.makedirs(SFT_CACHE_DIR, exist_ok=True)
        sft_train.save_to_disk(os.path.join(SFT_CACHE_DIR, "train"))
        sft_val.save_to_disk(os.path.join(SFT_CACHE_DIR, "val"))
        sft_test.save_to_disk(os.path.join(SFT_CACHE_DIR, "test"))

    print("✅ SFT sizes:", len(sft_train), len(sft_val), len(sft_test))
    # ==== END CACHE ====

    

    def left_pad_collator(features):
    # 取 batch 內的最長序列長度
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids  = f["input_ids"]
            mask = f["attention_mask"]
            lab  = f["labels"]

            pad = max_len - len(ids)
            # 左側補齊
            input_ids.append([tok.pad_token_id]*pad + ids)
            attention_mask.append([0]*pad + mask)
            labels.append([-100]*pad + lab)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
    collator = left_pad_collator

    # ---- 5) 訓練參數 ----
    class NoMoveTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.skipped_batches = 0

        def _move_model_to_device(self, model, device):
            # 我們自己管理 device_map（HF auto/shard），所以不要 Trainer 再搬動
            return model

        @staticmethod
        def _safe_zero(model):
            # 回傳「掛在參數圖上的 0」，讓 backward 有 grad_fn 但實際不更新
            for p in model.parameters():
                if p.requires_grad:
                    return p.sum() * 0.0
            # 萬一沒有可訓練參數（理論上不會），仍回一個需要梯度的 0
            dev = next(model.parameters()).device
            return torch.tensor(0.0, device=dev, requires_grad=True)

        def compute_loss(self, model, inputs, num_items_in_batch=None, **kwargs):
            # ① 先快速檢查：是否這個 batch 根本沒有可學的 label
            labels = inputs.get("labels", None)
            if labels is not None and (labels != -100).sum() == 0:
                self.skipped_batches += 1
                print("⚠️ batch has 0 valid labels; skipping.")
                # 清掉殘留 grad（保險）
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = None
                return self._safe_zero(model)

            # ② 正常 forward
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else getattr(outputs, "loss", None)

            # ③ 規範成 scalar tensor
            if loss is not None:
                loss = loss.mean()

            # ④ 檢查 NaN/Inf 或無梯度（例如某些實作回傳常數 0）
            bad = (loss is None) or (not torch.isfinite(loss)) or (not loss.requires_grad)
            if bad:
                self.skipped_batches += 1
                if loss is None:
                    print("⚠️ loss=None; skipping this batch.")
                elif not torch.isfinite(loss):
                    print("⚠️ non-finite loss detected; skipping this batch.")
                elif not loss.requires_grad:
                    print("⚠️ loss has no grad; skipping this batch.")
                # # 清掉殘留 grad，避免污染
                # for p in model.parameters():
                #     if p.grad is not None:
                #         p.grad = None
                return self._safe_zero(model)

            # ⑤ 正常回傳 loss（可被 AMP/Accelerate scale/backward）
            return loss




    from inspect import signature

    def make_training_args(**common):
        sig = signature(TrainingArguments.__init__).parameters
        # 過濾掉不支援的參數（避免老版 transformers 爆）
        common = {k: v for k, v in common.items() if k in sig}

        if "evaluation_strategy" in sig:
            common["evaluation_strategy"] = "no"   # ← 關掉訓練中的 eval
        elif "eval_strategy" in sig:
            common["eval_strategy"] = "no"
        elif "do_eval" in sig:
            common["do_eval"] = False

        # 🔒 關保存策略，避免 Trainer 自己存
        if "save_strategy" in sig:
            common["save_strategy"] = "no"
        # 不需要 save_steps
        common.pop("save_steps", None)

        return TrainingArguments(**common)

    # 共同參數（保持你原本設定）
    common_args = dict(
        output_dir="gptoss20b_lora_abs",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        max_steps=50, 
        learning_rate=2e-6,
        num_train_epochs=2,
        max_grad_norm=0.5,
        logging_steps=5,
        # save_steps=1000,
        weight_decay=0.01,
        warmup_ratio=0.10,
        lr_scheduler_type="cosine",
        bf16=False,   # V100
        fp16=True,    # V100 開 FP16
        gradient_checkpointing=True,      
        # evaluation_strategy="no",         # 沒有 evaluation strategy 參數
        # predict_with_generate=False, # 沒有 predict_with_generate 參數
        report_to="none",
        remove_unused_columns=False,
    )



    args = make_training_args(**common_args)
    print("✅ TrainingArguments constructed with:", args)
    

    # after args is constructed
    amp_dtype = torch.float16 if getattr(args, "fp16", False) else (torch.bfloat16 if getattr(args, "bf16", False) else None)
    amp_ctx = (torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype is not None else nullcontext())

    from contextlib import contextmanager

    @contextmanager
    def temporarily_enable_cache(model):
        prev = bool(getattr(model.config, "use_cache", False))
        model.config.use_cache = True
        try:
            yield
        finally:
            model.config.use_cache = prev


    dl = torch.utils.data.DataLoader(sft_train, batch_size=1, collate_fn=collator)

    chk = torch.utils.data.DataLoader(sft_train, batch_size=1, collate_fn=collator)
    for i, b in enumerate(chk):
        nvalid = (b["labels"] != -100).sum().item()
        if nvalid == 0:
            print(f"❗ batch {i} has 0 valid labels")
            break
        if i > 50:  # 看前 50 個就好
            break


    batch = next(iter(dl))

    def find_first_nan_module(model, batch, amp_ctx):
        bad = {"name": None}
        handles = []

        def make_hook(name):
            def hook(_, __, out):
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                    bad["name"] = name
                    raise RuntimeError(f"NaN in {name}")
            return hook

        for name, m in model.named_modules():
            if any(k in name.lower() for k in ["layer","block","mlp","experts","attention","router","norm"]):
                handles.append(m.register_forward_hook(make_hook(name)))

        try:
            with torch.no_grad(), amp_ctx:
                _ = model(
                    input_ids=batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                )
        except Exception as e:
            print("First NaN at module:", bad["name"], "|", e)
        finally:
            for h in handles:
                h.remove()

    # 呼叫
    find_first_nan_module(model, batch, amp_ctx)


    num_valid = (batch["labels"] != -100).sum().item()
    print("valid tokens in labels:", num_valid)
    assert num_valid > 0, "All labels are -100; loss would be undefined."

    # 可選：開 hidden states 來看是否前面就 NaN 了
    with torch.no_grad(), amp_ctx:
        # with amp_ctx:
        out_dbg = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            labels=None,
            output_hidden_states=True,
            return_dict=True,
        )
    hs = out_dbg.hidden_states[-1]
    print("last hidden finite?", torch.isfinite(hs).all().item())


    # ---- 6) ROUGE 評估（用生成）----
    rouge = load("rouge")

    def generate_text(batch):
        outs = []
        with temporarily_enable_cache(model):               # ← 這行
            for art in batch["article"]:
                prompt = build_prompt(art)
                ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(model.device)
                with torch.no_grad(), amp_ctx:              # ← 和訓練一致的 AMP
                    gen = model.generate(
                        **ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=0.0,
                        top_p=0.0,
                        do_sample=False,
                        eos_token_id=tok.eos_token_id,
                    )
                text = tok.decode(gen[0], skip_special_tokens=True)
                outs.append(text[len(prompt):].strip())
        return outs


    def compute_metrics_eval(eval_pred):
        sample_n = min(200, len(sft_val_raw))
        articles = [sft_val_raw[i]["article"] for i in range(sample_n)]
        refs     = [sft_val_raw[i]["abstract"] for i in range(sample_n)]

        preds = []
        with temporarily_enable_cache(model):               
            for art in articles:
                prompt = build_prompt(art)
                ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(model.device)
                with torch.no_grad(), amp_ctx:              
                    gen = model.generate(
                        **ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=0.0,
                        top_p=1.0,
                        do_sample=False,
                        eos_token_id=tok.eos_token_id,
                    )
                text = tok.decode(gen[0], skip_special_tokens=True)
                preds.append(text[len(prompt):].strip())

        scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
        scores["gen_len"] = sum(len(p.split()) for p in preds) / max(1, len(preds))
        return scores


    

    # try:
    #     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    # except TypeError:
    #     model.gradient_checkpointing_enable()

    # 4) 🛡️ 確保 router 沒隨機性（若這些屬性存在就設 0；不存在就忽略）
    for attr in ["router_dropout", "expert_dropout", "hidden_dropout", "attention_dropout", "embd_pdrop"]:
        if hasattr(model.config, attr):
            setattr(model.config, attr, 0.0)

    


    trainer = NoMoveTrainer(
        model=model,
        args=args,
        train_dataset=sft_train,
        eval_dataset=sft_val,
        data_collator=collator,
        tokenizer=tok,   
        # processing_class=tok,      # ← 取代 tokenizer=tok
        compute_metrics=compute_metrics_eval,
    )
    from torch.utils.data import DataLoader
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    test_loader = DataLoader(sft_train, batch_size=1, collate_fn=collator)
    batch = next(iter(test_loader))
    batch = {k: v.to(model.device) for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)
    print("quick sanity loss:", float(out.loss))
    # 不要手動 backward，交給 Trainer



    print("🚀 Start training gpt-oss-20b LoRA…")
    trainer.train()
    print("Skipped batches:", trainer.skipped_batches)


    save_dir = "gptoss20b_lora_abs"
    save_lora_adapters_skip_meta(model, save_dir)   
    # trainer.save_model("gptoss20b_lora_abs")


    @torch.no_grad()
    def batched_generate_preds_refs(ds, batch_size=2, max_items=None):
        N = len(ds) if max_items is None else min(len(ds), max_items)
        preds, refs = [], []
        with temporarily_enable_cache(model):               # ← 這行
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                articles   = [ds[i]["article"] for i in range(start, end)]
                references = [ds[i]["abstract"] for i in range(start, end)]

                prompts = [build_prompt(a) for a in articles]
                inputs = tok(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                ).to(model.device)

                with amp_ctx:                               # ← AMP
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                        eos_token_id=tok.eos_token_id,
                    )
                texts = tok.batch_decode(outputs, skip_special_tokens=True)

                for full, prompt, ref in zip(texts, prompts, references):
                    preds.append(full[len(prompt):].strip())
                    refs.append(ref)
        return preds, refs



    print("📏 Evaluating on pubmed_val (ROUGE-1/2/L)…")
    # 如需加快首次跑測可加 max_items=200；正式評估拿掉即可
    predictions, references = batched_generate_preds_refs(sft_val_raw, batch_size=2)

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    # evaluate 的 ROUGE 會回傳 0~1 的分數
    print(
        "ROUGE-1: {:.4f} | ROUGE-2: {:.4f} | ROUGE-L: {:.4f}".format(
            rouge_scores["rouge1"], rouge_scores["rouge2"], rouge_scores["rougeL"]
        )
    )



if __name__ == "__main__":
    main()