import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist
import math
random.seed(42)
import time

class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()})
        loss = unscaled_loss / t
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
        return loss if not return_outputs else (loss, outputs)


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
        batch["input_ids"] = noisy_batch.long()
        return batch


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""


SYSTEM_PROMPT_gsm8k_reasoning = """
Respond in the following format:
<thinking>
Your thinking here
</thinking>
<reasoning>
Your reasoning here
</reasoning>
<reflection>
Your reflection here
</reflection>
<adjustment>
Your adjustment here
</adjustment>
<output>
...
</output>
"""

SYSTEM_PROMPT_sudoku = """
Respond in the following format:
<answer>
...
</answer>
"""

def preprocess_dataset(data, tokenizer, max_length, test_split):
    preprocessed_data = []
    preprocessed_mia_data = []
    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        ## gsm8k: 
        # question = SYSTEM_PROMPT + "\n\n" + data[i]["question"] # gsm8k
        # prompt = [{"role": "user", "content": question}] # gsm8k
        # # trajectory = f"<answer>{data[i]['answer']}</answer>" # gsm8k
        # text = data[i]['answer']
        # part1, part2 = text.split("####", 1)
        # part1 = part1.strip()
        # part2 = part2.strip()
        # trajectory = f"<reasoning>{part1}</reasoning>\n<answer>{part2}</answer>" # gsm8k
        # response = [{"role": "assistant", "content": trajectory}] # gsm8k
        # response_mia = [{"role": "assistant", "content": ""}] #  gsm8k
        
        ## codealpaca
        question = data[i]["prompt"] # codealpaca
        prompt = [{"role": "user", "content": question}] # codealpaca
        response = [{"role": "assistant", "content": data[i]["completion"]}] # codealpaca
        response_mia = [{"role": "assistant", "content": ""}] # codealpaca

        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        inputs_mia = tokenizer.apply_chat_template(prompt + response_mia, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"


        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)
        tokenized_input_mia = tokenizer(
            inputs_mia, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)

        num_tokens = tokenized_input.shape[0]

        tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    

        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )
        preprocessed_mia_data.append(
            {
                "input_ids": tokenized_input_mia,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )

    combined = list(zip(preprocessed_data, preprocessed_mia_data))
    random.shuffle(combined)
    n_test = int(len(combined) * test_split)
    test_combined  = combined[:n_test]
    train_combined = combined[n_test:]
    test_data,      test_data_mia      = zip(*test_combined)
    train_data,    train_data_mia     = zip(*train_combined)

    new_test_size     = len(train_data)
    indices           = random.sample(range(len(test_data)), new_test_size)
    new_test_data     = [test_data[i]     for i in indices]
    new_test_data_mia = [test_data_mia[i] for i in indices]

    leftover      = [test_data[i]     for i in range(len(test_data))     if i not in indices]
    leftover_mia  = [test_data_mia[i] for i in range(len(test_data_mia)) if i not in indices]

    first_half_size = math.ceil(len(leftover) / 2)

    left_test_data      = leftover[:first_half_size]
    left_train_data     = leftover[first_half_size:]
    left_test_data_mia  = leftover_mia[:first_half_size]
    left_train_data_mia = leftover_mia[first_half_size:]


    return train_data, new_test_data, left_train_data, left_test_data,  \
        train_data_mia, new_test_data_mia, left_train_data_mia, left_test_data_mia
