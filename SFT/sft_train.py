import torch
import argparse
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import os
from sft_trainer import *
import torch.distributed as dist
import random
import numpy as np

def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Initialize argument parser
def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--model_name", type=str, default="Dream-org/Dream-v0-Instruct-7B", help="Name of the pretrained model"
    ) # Dream-org/Dream-v0-Instruct-7B, GSAI-ML/LLaDA-8B-Instruct
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum sequence length for tokenization"
    )
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--data_split_rate", type=float, default=0.7, help="test split rate")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="your_local_path/exps",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument("--job_name", type=str, default="dream-CodeAlpaca_20K-0.7", help="Job Name")
    parser.add_argument("--train_data", type=str, default="sahil2801/CodeAlpaca-20k", help="Path to training data") # openai/gsm8k, sahil2801/CodeAlpaca-20k
    parser.add_argument(
        "--debugging", action="store_true", help="Use while debugging model - only disables wandb logging"
    )

    return parser.parse_args()


def load_model_and_tokenizer(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )

    # Load model
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Applying LoRA model
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)  # Cast fp32 lora params to bf16
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_trainable}")

    return tokenizer, model

def load_saved_datasets(args, tokenizer):
    base = os.path.join(args.output_dir, args.job_name)

    train_data      = torch.load(os.path.join(base, "train_data.pt"))
    eval_data       = torch.load(os.path.join(base, "test_data.pt"))
    train_data_mia  = torch.load(os.path.join(base, "train_data_mia.pt"))
    eval_data_mia   = torch.load(os.path.join(base, "test_data_mia.pt"))
    left_train_data     = torch.load(os.path.join(base, "left_train_data.pt"))
    left_eval_data      = torch.load(os.path.join(base, "left_test_data.pt"))
    left_train_data_mia = torch.load(os.path.join(base, "left_train_data_mia.pt"))
    left_eval_data_mia  = torch.load(os.path.join(base, "left_test_data_mia.pt"))

    train_dataset       = dLLMSFTDataset(train_data,      tokenizer, args.max_length)
    eval_dataset        = dLLMSFTDataset(eval_data,       tokenizer, args.max_length, eval=True)
    train_dataset_mia   = dLLMSFTDataset(train_data_mia,  tokenizer, args.max_length)
    eval_dataset_mia    = dLLMSFTDataset(eval_data_mia,   tokenizer, args.max_length, eval=True)

    left_train_dataset     = dLLMSFTDataset(left_train_data,     tokenizer, args.max_length)
    left_eval_dataset      = dLLMSFTDataset(left_eval_data,      tokenizer, args.max_length, eval=True)
    left_train_dataset_mia = dLLMSFTDataset(left_train_data_mia, tokenizer, args.max_length)
    left_eval_dataset_mia  = dLLMSFTDataset(left_eval_data_mia,  tokenizer, args.max_length, eval=True)

    return (
        train_dataset,
        eval_dataset,
        train_dataset_mia,
        eval_dataset_mia,
        left_train_dataset,
        left_eval_dataset,
        left_train_dataset_mia,
        left_eval_dataset_mia
    )

def load_data(args, tokenizer):
    # data = load_dataset(args.train_data, "main", split="train") # gsm8k
    data = load_dataset(args.train_data, split="train") # codealpaca
    
    print("data: ", data)
    train_data, eval_data, left_train_data, left_eval_data, \
    train_data_mia, eval_data_mia, left_train_data_mia, left_eval_data_mia = preprocess_dataset(data, tokenizer, args.max_length, args.data_split_rate)
    output_dir = os.path.join(args.output_dir, args.job_name)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, args.job_name, "train_data.pt")
    eval_path  = os.path.join(args.output_dir, args.job_name, "test_data.pt")
    torch.save(train_data, train_path)
    torch.save(eval_data, eval_path)

    train_path_mia = os.path.join(args.output_dir, args.job_name, "train_data_mia.pt")
    eval_path_mia  = os.path.join(args.output_dir, args.job_name, "test_data_mia.pt")
    torch.save(train_data_mia, train_path_mia)
    torch.save(eval_data_mia, eval_path_mia)

    left_train_path = os.path.join(args.output_dir, args.job_name, "left_train_data.pt")
    left_eval_path  = os.path.join(args.output_dir, args.job_name, "left_test_data.pt")
    torch.save(left_train_data, left_train_path)
    torch.save(left_eval_data, left_eval_path)

    left_train_path_mia = os.path.join(args.output_dir, args.job_name, "left_train_data_mia.pt")
    left_eval_path_mia  = os.path.join(args.output_dir, args.job_name, "left_test_data_mia.pt")
    torch.save(left_train_data_mia, left_train_path_mia)
    torch.save(left_eval_data_mia, left_eval_path_mia)
    print(f"Data Processed and Saved")


    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)

    left_train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    left_eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)

    return train_dataset, eval_dataset, left_train_dataset, left_eval_dataset


# Training setup
def train_model(args, tokenizer, model):
    # Load dataset
    train_dataset, eval_dataset, left_train_dataset, left_eval_dataset = load_data(args, tokenizer)
    # train_dataset, eval_dataset, train_dataset_mia, \
    #     eval_dataset_mia, _, _, _, _  = load_saved_datasets(args, tokenizer)

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=2,
        save_steps=100,
        save_total_limit=20,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        report_to="wandb" if not args.debugging else "none",
        remove_unused_columns=False,
    )

    # Create optimizer and scheduler
    num_train_steps = int(
        len(train_dataset)
        * args.num_epochs
        / (args.batch_size * args.grad_accum_steps * torch.cuda.device_count())
    )
    # Initialize Trainer with custom dLLMTrainer
    trainer = dLLMTrainer(
        model=model,
        args=training_args,                                             # dream: 151666, llada: 126336 
        data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=151666, max_length=args.max_length),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    del trainer                   
    del model                  
    import gc
    gc.collect()  
    torch.cuda.empty_cache()
    torch.cuda.synchronize()



def train_left_model(args, tokenizer, model):
    # Load dataset
    _, _, _, _, left_train_dataset,left_eval_dataset,\
        left_train_dataset_mia,left_eval_dataset_mia = load_saved_datasets(args, tokenizer)

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name, "shadow"),
        num_train_epochs=args.num_epochs*3,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=2,
        save_steps=100,
        save_total_limit=20,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        report_to="wandb" if not args.debugging else "none",
        remove_unused_columns=False,
    )

    # Create optimizer and scheduler
    num_train_steps = int(
        len(left_train_dataset)
        * args.num_epochs
        / (args.batch_size * args.grad_accum_steps * torch.cuda.device_count())
    )
    # Initialize Trainer with custom dLLMTrainer
    trainer = dLLMTrainer(
        model=model,
        args=training_args,                                             # dream: 151666, llada: 126336 
        data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=151666, max_length=args.max_length),
        train_dataset=left_train_dataset,
        eval_dataset=left_eval_dataset,
    )

    # Start training
    trainer.train()
    del trainer                    
    del model                        

if __name__ == "__main__":
    init_seed(42)
    args = parse_args()
    tokenizer, model = load_model_and_tokenizer(args)
    train_model(args, tokenizer, model)

    # left_tokenizer, left_model = load_model_and_tokenizer(args)
    # train_left_model(args, left_tokenizer, left_model)



