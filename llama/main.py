import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

from data import data_prep
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from classes import EvalEveryNStepsCallback

def main():
    
    MAX_SEQ_LENGTH = 2048
    
    ## ----------model-----------------##

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "models/unsloth-llama-3.2-3b",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )


    

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )
    ## --------------------------------##

    ## --------data preparation -------##
    train_dataset, val_dataset, test_dataset = data_prep(tokenizer)

    ## --------------------------------##

    ## --------training----------------##
    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        warmup_steps=5,
        learning_rate=2e-5,
        logging_steps=50,

        save_strategy="steps",     # 仍然每 N 步存檔
        save_steps=1000,
        save_total_limit=2,
        save_safetensors=True,

        # 這些在你的環境有支援
        load_best_model_at_end=False,   # 我們自己處理「最佳模型」，先關掉內建
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,

        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_num_proc=1,

        args=args,
        callbacks=[EvalEveryNStepsCallback(eval_steps=1000, metric="eval_loss", greater_is_better=False)],
    )


    trainer_stats = trainer.train()

    ## ----------------------------------##

    ## ----------save--------------------##
    model.save_pretrained("lora_model")  # Local saving
    tokenizer.save_pretrained("lora_model")
    ## ----------------------------------##
    
    
    return 


if __name__ == "__main__":
    main()