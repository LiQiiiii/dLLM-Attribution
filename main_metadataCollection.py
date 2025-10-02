import torch
import numpy as np
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel
import argparse
import re

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@ torch.no_grad()
def generate(model, prompt, tokenizer, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='random', mask_id=126336):
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    binary_list = []
    confidence_list = []
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0_p_response = x0_p[:, prompt.shape[1]:].clone()
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            print("x: ", tokenizer.batch_decode(x, skip_special_tokens=True)[0])
            x_response = x[:, prompt.shape[1]:]
            mask_id = 126336
            binary_tensor = (x_response != mask_id).long().squeeze(0)   # shape (512,)
            confidence_list.append(x0_p_response)
            binary_list.append(binary_tensor)
            print(f"block {num_block} ep {i:3d}", binary_tensor.sum().item(), f"/ {gen_length} FILLED")
    all_binary = torch.stack(binary_list) 
    confidence_list = [t.squeeze() for t in confidence_list]
    all_confidence = torch.stack(confidence_list)
    filtered_confidence = torch.where(all_binary.bool(), all_confidence, torch.zeros_like(all_confidence))
    return x, all_binary, all_confidence, filtered_confidence

def parse_args():
    parser = argparse.ArgumentParser(description="Run attribution metadata collection for dLLMs.")

    # experiment settings
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Name of the experiment (used for saving checkpoints and results).")
    parser.add_argument("--non_exp_name", type=str, default=None,
                        help="Name for non-target experiment (default: same as exp_name).")
    parser.add_argument("--ckpt_num", type=int, required=True,
                        help="Checkpoint number used to collect attribution metadata.")

    # model and training settings
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Model name or path (e.g., GSAI-ML/LLaDA-8B-Instruct, Dream-org/Dream-v0-Instruct-7B).")
    parser.add_argument("--mask_token_id", type=int, default=126336,
                        help="Mask token id (dream: 151666, llada: 126336).")
    parser.add_argument("--gen_length", type=int, default=32,
                        help="Generation length (steps).")
    parser.add_argument("--block_size", type=int, default=32,
                        help="Block size for generation.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on (e.g., cuda:0, cuda:7).")

    return parser.parse_args()

def main():
    args = parse_args()
    if args.non_exp_name is None:
        args.non_exp_name = args.exp_name

    peft_path = f"./SFT/exps/{args.exp_name}/checkpoint-{args.ckpt_num}"
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True) 
    
    model = PeftModel.from_pretrained(model, peft_path, torch_dtype=torch.bfloat16).to(args.device).eval()

    train_path = f"./SFT/exps/{args.exp_name}/train_data_mia.pt"
    eval_path = f"./SFT/exps/{args.exp_name}/test_data_mia.pt"

    train_data = torch.load(train_path)
    eval_data  = torch.load(eval_path)

    target_history, target_confidence, target_filtered = [], [], [], [], [], []
    local_history, local_confidence, local_filtered = [], [], [], [], [], []
    target_effect, local_effect = [], [], [], []

    print("len(train_data): ", len(train_data))
    print("len(eval_data): ", len(eval_data))

    for mm in range(len(train_data)):
        print(f"tar train - {mm}")
        input_ids = train_data[mm]['input_ids']
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        if args.model_name == "Dream-org/Dream-v0-Instruct-7B":
            pattern = r"<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>\s*<\|im_start\|>assistant"
            match = re.search(pattern, text, re.DOTALL)
            text = match.group(1).strip()
        elif args.model_name == "GSAI-ML/LLaDA-8B-Instruct":
            if text.startswith("user"):
                text = text[len("user"):].lstrip()
            parts = text.rsplit("assistant", 2)  
            text = parts[0].rstrip()
        messages = [{"role": "user", "content": text}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        input_ids = inputs.input_ids.to(device="cuda")

        output, history_trajectory, confidence_trajectory, filtered_confidence_trajectory = generate(model, input_ids, tokenizer, steps=gen_length, gen_length=gen_length, block_length=block_size, temperature=0., cfg_scale=0., remasking='low_confidence')
        unmask_confidence = torch.where(history_trajectory.bool(), confidence_trajectory, torch.full_like(confidence_trajectory, -1.0))
        mask_confidence = torch.where(history_trajectory.bool(), torch.full_like(confidence_trajectory, -1.0), confidence_trajectory)
        unmask_confidence_np = unmask_confidence.to(torch.float32).cpu().numpy()
        T, L = unmask_confidence_np.shape
        effect_map = np.zeros((T, L), dtype=np.float32)
        for i in range(1, T):
            prev_conf = unmask_confidence_np[i - 1]
            curr_conf = unmask_confidence_np[i]
            delta = curr_conf - prev_conf
            newly_unmasked = (prev_conf == -1) & (curr_conf != -1)
            already_unmasked = (prev_conf != -1) & (curr_conf != -1)
            for k in np.where(newly_unmasked)[0]:
                effect_type = 0
                for m in np.where(already_unmasked)[0]:
                    if delta[m] < 0:  
                        effect_map[i, k] = 10 if effect_map[i, k] == 0.5 else -0.5 if effect_map[i, k] == 0 else -0.5
                        effect_map[i, m] = -2
                        effect_type = -1
                    elif delta[m] > 0:  
                        effect_map[i, k] = 10 if effect_map[i, k] == -0.5 else 0.5 if effect_map[i, k] == 0 else 0.5
                        effect_map[i, m] = 2
                        effect_type = 1
                if effect_type == 0:
                    effect_map[i, k] = 0
        effect_map_tensor = torch.from_numpy(effect_map).float()
        target_effect.append(effect_map_tensor)
        target_history.append(history_trajectory)
        target_confidence.append(confidence_trajectory)
        target_filtered.append(filtered_confidence_trajectory)   

    torch.save(target_history, f"./SFT/exps/{args.exp_name}/attribute_target_history_gen{args.gen_length}_block{args.block_size}.pt")
    torch.save(target_confidence, f"./SFT/exps/{args.exp_name}/attribute_target_confidence_gen{args.gen_length}_block{args.block_size}.pt")  
    torch.save(target_filtered, f"./SFT/exps/{args.exp_name}/attribute_target_filtered_gen{args.gen_length}_block{args.block_size}.pt")  
    torch.save(target_effect, f"./SFT/exps/{args.exp_name}/attribute_target_effect_gen{args.gen_length}_block{args.block_size}.pt") 

    for mm in range(len(eval_data)):
        print(f"tar eval - {mm}")
        input_ids = eval_data[mm]['input_ids']
        
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        if args.model_name == "Dream-org/Dream-v0-Instruct-7B":
            pattern = r"<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>\s*<\|im_start\|>assistant"
            match = re.search(pattern, text, re.DOTALL)
            text = match.group(1).strip()
        elif args.model_name == "GSAI-ML/LLaDA-8B-Instruct":
            if text.startswith("user"):
                text = text[len("user"):].lstrip()
            parts = text.rsplit("assistant", 2)  
            text = parts[0].rstrip()
        messages = [{"role": "user", "content": text}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        input_ids = inputs.input_ids.to(device="cuda")

        output, history_trajectory, confidence_trajectory, filtered_confidence_trajectory = generate(model, input_ids, tokenizer, steps=gen_length, gen_length=gen_length, block_length=block_size, temperature=0., cfg_scale=0., remasking='low_confidence')
        
        unmask_confidence = torch.where(history_trajectory.bool(), confidence_trajectory, torch.full_like(confidence_trajectory, -1.0))
        unmask_confidence_np = unmask_confidence.to(torch.float32).cpu().numpy()
        T, L = unmask_confidence_np.shape
        effect_map = np.zeros((T, L), dtype=np.float32)
        for i in range(1, T):
            prev_conf = unmask_confidence_np[i - 1]
            curr_conf = unmask_confidence_np[i]
            delta = curr_conf - prev_conf
            newly_unmasked = (prev_conf == -1) & (curr_conf != -1)
            already_unmasked = (prev_conf != -1) & (curr_conf != -1)
            for k in np.where(newly_unmasked)[0]:
                effect_type = 0
                for m in np.where(already_unmasked)[0]:
                    if delta[m] < 0:  
                        effect_map[i, k] = 10 if effect_map[i, k] == 0.5 else -0.5 if effect_map[i, k] == 0 else -0.5
                        effect_map[i, m] = -2
                        effect_type = -1
                    elif delta[m] > 0:  
                        effect_map[i, k] = 10 if effect_map[i, k] == -0.5 else 0.5 if effect_map[i, k] == 0 else 0.5
                        effect_map[i, m] = 2
                        effect_type = 1
                if effect_type == 0:
                    effect_map[i, k] = 0
        effect_map_tensor = torch.from_numpy(effect_map).float()
        local_effect.append(effect_map_tensor)

        local_history.append(history_trajectory)
        local_confidence.append(confidence_trajectory)
        local_filtered.append(filtered_confidence_trajectory)   

    torch.save(local_history, f"./SFT/exps/{args.exp_name}/attribute_targetnon_history_gen{args.gen_length}_block{args.block_size}.pt")
    torch.save(local_confidence, f"./SFT/exps/{args.exp_name}/attribute_targetnon_confidence_gen{args.gen_length}_block{args.block_size}.pt")  
    torch.save(local_filtered, f"./SFT/exps/{args.exp_name}/attribute_targetnon_filtered_gen{args.gen_length}_block{args.block_size}.pt")  
    torch.save(local_effect, f"./SFT/exps/{args.exp_name}/attribute_targetnon_effect_gen{args.gen_length}_block{args.block_size}.pt")  

if __name__ == '__main__':
    main()
