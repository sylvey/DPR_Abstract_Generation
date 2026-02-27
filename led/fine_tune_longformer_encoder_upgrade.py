import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import os
import json, time
from pathlib import Path
import pickle
#!/usr/bin/env python3
from datasets import load_dataset, load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch
from transformers import DataCollatorForSeq2Seq

from evaluate import load
from transformers import LEDTokenizer, LEDForConditionalGeneration
import shutil

from transformers import Seq2SeqTrainer

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
from torch.cuda.amp import autocast

from contextlib import nullcontext

from transformers import Seq2SeqTrainer
import torch
from torch.cuda.amp import autocast

from transformers import Seq2SeqTrainer
import torch

from transformers.trainer_callback import TrainerCallback



class IgnoreGradException(Exception):
    pass

class IgnoreGradCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(state, "log_history") and isinstance(state.log_history, list):
            if state.log_history and isinstance(state.log_history[-1], dict):
                state.log_history[-1]["skipped_batch"] = True

from transformers.models.led.modeling_led import LEDForConditionalGeneration

def generate_answer(batch):
    try:
        with torch.no_grad():
            inputs_dict = tokenizer(
                batch["article"],
                padding="max_length",
                max_length=8192,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs_dict["input_ids"].to("cuda")
            attention_mask = inputs_dict["attention_mask"].to("cuda")

            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1

            # 以下整段容易 OOM，要 catch
            predicted_abstract_ids = led.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                max_length=512,
                num_beams=4,
            )

            decoded = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
            batch["predicted_abstract"] = decoded

    except torch.cuda.OutOfMemoryError:
        print("⚠️ Skipping example due to OOM in generate_answer.")
        torch.cuda.empty_cache()
        batch["predicted_abstract"] = ["[OOM ERROR]"] * len(batch["article"])

    except RuntimeError as e:
        print(f"⚠️ RuntimeError in generate_answer: {e}")
        torch.cuda.empty_cache()
        batch["predicted_abstract"] = ["[Runtime ERROR]"] * len(batch["article"])

    return batch


class LEDForConditionalGenerationWithGlobalMask(LEDForConditionalGeneration):
    def forward(self, *args, **kwargs):
        # 直接攔截 global_attention_mask 不存在的情況，補上 default
        if "global_attention_mask" not in kwargs and "attention_mask" in kwargs:
            attn = kwargs["attention_mask"]
            kwargs["global_attention_mask"] = torch.zeros_like(attn)
            kwargs["global_attention_mask"][:, 0] = 1

        return super().forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs.pop("labels", None)
        return super().generate(*args, **kwargs)
        
class SafeTrainer(Seq2SeqTrainer):
    def training_step(self, model, inputs):
        print("🔥 Actually running a training step...")
        model.train()
        inputs = self._prepare_inputs(inputs)

        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            # NaN/Inf 防護
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("NaN or Inf loss detected.")

            # 多卡平均
            if self.args.n_gpu > 1 and hasattr(loss, "mean"):
                loss = loss.mean()

            # ✅ 關鍵：用 accelerator/backward，讓 GradScaler 生效
            self.accelerator.backward(loss)

            # HF Trainer 慣例：回傳 detach 後的 loss
            return loss.detach()

        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
            print(f"⚠️ Skipping batch due to error: {e}")
            torch.cuda.empty_cache()

            # ✅ 維持 GradScaler 狀態：對「0-loss」做一次 backward（不產生梯度）
            dummy = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
            self.accelerator.backward(dummy)

            # 回傳 0，讓訓練流程不中斷
            return dummy.detach()


import torch
import torch.nn.functional as F

# 只做一次 monkey-patch，並把原始 softmax 綁成預設參數，避免後續被覆蓋
if not hasattr(F, "_orig_softmax"):
    F._orig_softmax = F.softmax

@torch.no_grad()
def stable_softmax(input, dim=None, **kwargs):
    # 先轉成 float32 做 softmax，再轉回原 dtype
    x = input.to(torch.float32)
    # 將 NaN/Inf 處理掉，避免傳到 softmax
    x = torch.nan_to_num(x, nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf"))
    out = F._orig_softmax(x, dim=dim, **kwargs)   # 用備份，不會遞迴
    return out.to(input.dtype)

F.softmax = stable_softmax

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    # 載入資料
    global tokenizer, led 

    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cache_path = "../data/pubmed_datasets_new.pkl"

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

        
        # pubmed_train = pubmed_train.select(range(min(len(pubmed_train), 8000)))
        # pubmed_val   = pubmed_val.select(range(min(len(pubmed_val), 800)))
        # pubmed_test  = pubmed_test.select(range(min(len(pubmed_test), 800)))

        tokenizer = AutoTokenizer.from_pretrained("./led_pubmed_model")
        encoder_max_length = 8192
        decoder_max_length = 512

        def process_data_to_model_inputs(example):
            inputs = tokenizer(
                example["article"],
                padding=False,
                truncation=True,
                max_length=encoder_max_length,
            )
            outputs = tokenizer(
                example["abstract"],
                padding=False,
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

    pubmed_train = pubmed_train.flatten_indices()
    pubmed_val = pubmed_val.flatten_indices()
    pubmed_test = pubmed_test.flatten_indices()

    # 再強制重建連續視圖（保險，避免殘留間接索引）
    pubmed_train = pubmed_train.select(range(len(pubmed_train)))
    pubmed_val   = pubmed_val.select(range(len(pubmed_val)))
    pubmed_test   = pubmed_test.select(range(len(pubmed_test)))


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./led_pubmed_model")

    # max encoder length is 8192 for PubMed
    encoder_max_length = 8192
    decoder_max_length = 512
    batch_size = 1

    eval_dataset_small = pubmed_val.select(range(min(len(pubmed_val), 100)))

    print(len(pubmed_train))

    # enable fp16 apex training
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        fp16=True,
        output_dir="./led_pubmed_finetune",
        logging_steps=250,
        eval_steps=500,                 # ← 讓訓練中真的會 eval
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,     # ← 重要
        metric_for_best_model="eval_rouge2_fmeasure",
        greater_is_better=True,
        max_grad_norm=0.5,          # originally 1.0
        warmup_ratio=0.06,               # ← 用比例，避免不同資料量失衡
        gradient_accumulation_steps=8,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        group_by_length=False,
        dataloader_drop_last=True,

        learning_rate=2e-5, # added for longer prompts

        # max_steps=11500,
        num_train_epochs=1,

        # 明確鎖住生成參數，避免路徑差異
        generation_max_length=128,
        generation_num_beams=1,
    )



    rouge = load("rouge")
    # compute Rouge score during validation
    def compute_metrics(eval_preds):

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # decode
        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.array(labels)  # 保證 ndarray
        labels = labels.copy()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
        labels[labels == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 先嘗試非聚合（理想可拿到 P/R/F）
        try:
            out = rouge.compute(
                predictions=pred_str,
                references=label_str,
                rouge_types=["rouge2"],
                use_aggregator=False,
            )["rouge2"]

            # list[Score]：平均 P/R/F
            if isinstance(out, list) and len(out) > 0 and hasattr(out[0], "precision"):
                p = float(np.mean([s.precision for s in out]))
                r = float(np.mean([s.recall    for s in out]))
                f = float(np.mean([s.fmeasure  for s in out]))
                return {
                    "rouge2_precision": round(p, 4),
                    "rouge2_recall":    round(r, 4),
                    "rouge2_fmeasure":  round(f, 4),
                }

            # list/ndarray[float]：當成 F1 平均
            if isinstance(out, (list, tuple, np.ndarray)):
                f = float(np.mean(out))
                return {
                    "rouge2_precision": round(f, 4),
                    "rouge2_recall":    round(f, 4),
                    "rouge2_fmeasure":  round(f, 4),
                }

            # 單一 float
            if isinstance(out, (float, np.floating)):
                f = float(out)
                return {
                    "rouge2_precision": round(f, 4),
                    "rouge2_recall":    round(f, 4),
                    "rouge2_fmeasure":  round(f, 4),
                }
        except Exception:
            pass

        # 退而求其次：用聚合（多版本直接回 F1 float，或回帶 .mid 的物件）
        agg = rouge.compute(
            predictions=pred_str,
            references=label_str,
            rouge_types=["rouge2"],
            use_aggregator=True,
        )["rouge2"]

        if isinstance(agg, (float, np.floating)):
            f = float(agg)
            return {
                "rouge2_precision": round(f, 4),
                "rouge2_recall":    round(f, 4),
                "rouge2_fmeasure":  round(f, 4),
            }

        # 物件路徑（帶 .mid）
        try:
            p = float(agg.mid.precision)
            r = float(agg.mid.recall)
            f = float(agg.mid.fmeasure)
        except Exception:
            # 萬一是其它結構，就盡力取 F1
            f = float(getattr(agg, "fmeasure", getattr(agg, "f1", 0.0)))
            p = float(getattr(agg, "precision", f))
            r = float(getattr(agg, "recall", f))

        return {
            "rouge2_precision": round(p, 4),
            "rouge2_recall":    round(r, 4),
            "rouge2_fmeasure":  round(f, 4),
        }


    # load model + enable gradient checkpointing & disable cache for checkpointing
    led = LEDForConditionalGenerationWithGlobalMask.from_pretrained(
        "./led_pubmed_model",
        use_cache=False,
        # torch_dtype=torch.float16,
    )


    led.gradient_checkpointing_enable()
    led.led.decoder.gradient_checkpointing = False
    # led.led.encoder.gradient_checkpointing = True
    # led.led.encoder._gradient_checkpointing = False
    # led.led.decoder.gradient_checkpointing = False
    # led.led.decoder._gradient_checkpointing = False





    # set generate hyperparameters
    led.config.num_beams = 4
    led.config.max_length = 512
    led.config.min_length = 100
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=led, label_pad_token_id=-100)

    # instantiate trainer
    trainer = SafeTrainer(
        model=led,
        args=training_args,
        train_dataset=pubmed_train,
        eval_dataset=eval_dataset_small,
        tokenizer=tokenizer,
        callbacks=[IgnoreGradCallback()],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Trainer train_dataset size:", len(trainer.train_dataset)) 

    # ⬇️ 把這段放在這裡（train 之前）
    dl = trainer.get_train_dataloader()
    num_batches = len(dl)  # drop_last=True 已生效
    updates_per_epoch = (num_batches + training_args.gradient_accumulation_steps - 1) // training_args.gradient_accumulation_steps
    total_steps = updates_per_epoch * int(training_args.num_train_epochs)
    print(f"num_batches={num_batches}, updates_per_epoch={updates_per_epoch}, total_steps={total_steps}")
    print(f"current_global_step={trainer.state.global_step}")


    # start training
    # add this for confirmation

    # import os

    checkpoint_dir = training_args.output_dir

    def extract_step(path):
        name = os.path.basename(path)
        try:
            return int(name.replace("checkpoint-", ""))
        except:
            return -1

    # 找出所有 checkpoint 子資料夾
    # 如果有 checkpoint，就 resume；否則從頭開始
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = [
            os.path.join(training_args.output_dir, d)
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint")
        ]
        if checkpoints:
            def extract_step(path):
                name = os.path.basename(path)
                try:
                    return int(name.replace("checkpoint-", ""))
                except:
                    return -1
            last_checkpoint = max(checkpoints, key=extract_step)
            print(f"⏪ Resuming from checkpoint: {last_checkpoint}")
        else:
            print("🚀 Starting training from scratch.")
    else:
        print("🚀 No checkpoint directory found. Starting from scratch.")


    try:
        if last_checkpoint is not None:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()
    except IgnoreGradException:
        print("⚠️ Skipped a batch but training continued.")
        


    # 列出所有 checkpoint
    checkpoints = [
        os.path.join(training_args.output_dir, d)
        for d in os.listdir(training_args.output_dir)
        if d.startswith("checkpoint")
    ]

    # 根據步數排序（由小到大）
    checkpoints_sorted = sorted(checkpoints, key=extract_step)

    # 只保留最後兩個
    to_delete = checkpoints_sorted[:-2]

    for path in to_delete:
        print(f"🧹 Removing old checkpoint: {path}")
        shutil.rmtree(path)


    
    led.to("cuda")
    led.eval()

    pubmed_test.set_format(type="python")

    result = pubmed_test.map(generate_answer, batched=True, batch_size=1)

    # ===== SAVE ALL TEST GENERATIONS =====
    out_dir = Path("./led_pubmed_finetune/preds")
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    gen_path = out_dir / f"test_generations_{stamp}.jsonl"       # main output
    meta_path = out_dir / f"test_generations_{stamp}.meta.json"  # sidecar metadata

    # Write JSONL: one example per line with id / pred / ref
    with gen_path.open("w", encoding="utf-8") as f:
        for i, (p, r) in enumerate(zip(result["predicted_abstract"], result["abstract"])):
            rec = {
                "id": i,
                "prediction": p,
                "reference": r,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Optional: record exact generation settings for provenance/repro
    meta = {
        "time": stamp,
        "model_name_or_path": "./led_pubmed_model",
        "num_beams": getattr(led.config, "num_beams", None),
        "max_length": getattr(led.config, "max_length", None),
        "min_length": getattr(led.config, "min_length", None),
        "length_penalty": getattr(led.config, "length_penalty", None),
        "early_stopping": getattr(led.config, "early_stopping", None),
        "no_repeat_ngram_size": getattr(led.config, "no_repeat_ngram_size", None),
        "tokenizer": tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "unknown",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved generations to: {gen_path}")
    print(f"📝 Saved metadata to:    {meta_path}")

    

    

if __name__ == "__main__":
    main()