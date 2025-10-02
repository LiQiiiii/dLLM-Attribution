from transformers import AutoModel,AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", padding_side="right", trust_remote_code=True, use_fast=True
    )
base_model = base_model.to(device)
from peft import PeftModel

adapter_checkpoint = "/home/liqi/LLM-Adapters/d1/SFT/exps/llada-sft-sudoku-0.5/checkpoint-2700"
model = PeftModel.from_pretrained(
    base_model,
    adapter_checkpoint,
    torch_dtype=torch.float16,
    local_files_only=True
)
print("model: \n", model)
