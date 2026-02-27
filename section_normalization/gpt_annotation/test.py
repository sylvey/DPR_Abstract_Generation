import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import PeftModel

from evaluate import load as load_metric
from data import load
from utilities import batched_generate
from trl import SFTTrainer, SFTConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from pathlib import Path

def main():
    
    ## ----------model-----------------##

    model_name = "../../gpt/gpt-oss-20b"
 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    tokenizer.chat_template = Path("../../gpt/gpt-oss-20b/chat_template.jinja").read_text(encoding="utf-8")
    _dbg = tokenizer.apply_chat_template([{"role":"user","content":"ping"}], add_generation_prompt=True, tokenize=False)
    print(_dbg[-80:]) 

    # MAX_SEQ_LENGTH = 512
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = model_name,
    #     max_seq_length = MAX_SEQ_LENGTH,
    #     dtype = None,           # 讓 unsloth 自己決定
    #     load_in_4bit = True,    # <--- 這就是魔法發生的地方！
    # )

    # unsloth 會自動處理 FastAttention 等優化
    # 不需要再手動設定 device_map="auto"，unsloth 會處理好 # 切换到評估模式  
        
    model.eval()
   
    ## --------------------------------##

    ## --------data preparation -------##
    _, _, test_set = load()
    
    batched_generate(test_set, model, tokenizer)
    

    return



if __name__ == "__main__":
    main()