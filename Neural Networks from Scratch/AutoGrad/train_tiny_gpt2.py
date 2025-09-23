import os
import argparse
import numpy as np
import cupy as cp
import requests
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from models.gpt2 import GPT2
from tqdm import tqdm

from transformers import GPT2TokenizerFast
import wandb

def main(args):

    ### MODEL PARAMETERS ###
    CONTEXT_LENGTH = args.context_length
    EMBED_DIM = args.embed_dim
    NUM_HEADS = args.num_heads
    NUM_BLOCKS = args.num_blocks
    DROPOUT_P = args.dropout
    MLP_RATIO = args.mlp_ratio
    PATH_TO_DATA = args.data_path
    USE_GPT2_TOKENIZER = args.use_gpt2_tokenizer

    ### TRAINING PARAMETERS ###
    TRAINING_ITERATIONS = args.train_iterations
    BATCH_SIZE = args.batch_size
    MAX_LEARNING_RATE = args.max_lr
    MIN_LEARNING_RATE = args.min_lr
    WARMUP_STEPS = args.warmup_steps
    WEIGHT_DECAY = args.weight_decay
    MAX_GRAD_NORM = args.max_grad_norm
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation

    ### LOGGING PARAMETERS ###
    LOG_ITER = args.log_iter
    GEN_ITER = args.gen_iter
    SAMPLE_GEN_SEED = "h"
    GEN_LENGTH = args.gen_length
    LOG_WANDB = args.log_wandb

    ### SAVE PATH ###
    SAVE_PATH = args.save_path

    ### INITIALIZE WANDB ###
    if LOG_WANDB:
        wandb.init(
            project="autograd_tiny_gpt2_harry_potter",
            name="mini_gpt2_run",
            config={
                "context_length": CONTEXT_LENGTH,
                "embed_dim": EMBED_DIM,
                "num_heads": NUM_HEADS,
                "num_blocks": NUM_BLOCKS,
                "dropout": DROPOUT_P,
                "batch_size": BATCH_SIZE,
                "learning_rate": MAX_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY
            }
        )

    ### PREPARE DATASET ###
    def get_text_data(path_to_data):

        ### Load Single Text File ###
        if ".txt" in path_to_data:
            
            ### If a link ###
            if "https://" in path_to_data:
                all_txt = requests.get(path_to_data).text
            else:
                with open(path_to_data, "r") as f:
                    all_txt = f.readlines(f)
        else:
            files = [os.path.join(path_to_data, file) for file in os.listdir(path_to_data) if ".txt" in file]
            all_txt = ""
            for file in files:
                with open(file, "r") as f:
                    text = f.readlines()

                ### Some basic cleanup of the harry potter text ###
                text = [line for line in text if "Page" not in line]
                text = " ".join(text).replace("\n", "")
                text = [word for word in text.split(" ") if len(word) > 0]
                text = " ".join(text)
                all_txt+=text

                break
            
        return all_txt

    all_text = get_text_data(PATH_TO_DATA)

    ### Tokenize Text ###
    if USE_GPT2_TOKENIZER:
        print("Using GPT2 Tokenizer")
        tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
        tokens = cp.array(tokenizer.encode(all_text))
    else:
        print("Training Character Level Model")
        chars = sorted(list(set(all_text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}

        class Tokenizer:
            def __init__(self, char2idx, idx2char):
                self.char2idx = char2idx
                self.idx2char = idx2char
                self.vocab_size = len(char2idx)
            
            def encode(self, input):
                return [self.char2idx[i] for i in input]

            def decode(self, input):
                return "".join([self.idx2char[i] for i in input])
        
        tokenizer = Tokenizer(char_to_idx, idx_to_char)
        tokens = cp.array(tokenizer.encode(all_text))

    ### Grab Batches from Dataset ###
    def get_batch(data, batch_size, seq_len):
        idx = cp.random.randint(0, len(data) - seq_len - 1, batch_size)
        inputs = cp.stack([data[i:i+seq_len] for i in idx])
        targets = cp.stack([data[i+1:i+seq_len+1] for i in idx])
        return mytorch.Tensor(inputs), mytorch.Tensor(targets)

    ### Load Model ###
    model = GPT2(vocab_size=tokenizer.vocab_size, 
                max_seq_len=CONTEXT_LENGTH, 
                embed_dim=EMBED_DIM, 
                num_heads=NUM_HEADS, 
                num_blocks=NUM_BLOCKS, 
                dropout_p=DROPOUT_P, 
                mlp_ratio=MLP_RATIO)
    
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += np.prod(param.shape)
    print("Total Trainable Parameters:", total_params)

    ### Load Causal Mask for Training ###
    causal_mask = mytorch.Tensor(
        cp.triu(cp.ones((1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH)) * float('-inf'), k=1)
    ).astype(cp.float32)

    ### Load Optimizer ###
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    ### Load Scheduler ###
    scheduler = mytorch.lr_scheduler.CosineLRScheduler(
        optimizer=optimizer, max_lr=MAX_LEARNING_RATE, 
        min_lr=MIN_LEARNING_RATE, total_steps=TRAINING_ITERATIONS,
        warmup_steps=WARMUP_STEPS
    )

    ### Load Loss Function ###
    loss_fn = nn.CrossEntropyLoss()

    ### Train Model ###
    def train_step():

        for i in range(GRADIENT_ACCUMULATION_STEPS):

            inputs, targets = get_batch(tokens, BATCH_SIZE//GRADIENT_ACCUMULATION_STEPS, CONTEXT_LENGTH)
        
            ### Compute Logits ###

            logits = model(inputs, causal_mask)
    
            ### Compute Loss ###
            loss = loss_fn(logits, targets) / GRADIENT_ACCUMULATION_STEPS

            ### Compute Gradients ###
            loss.backward()
        
        ### Clip Gradients ###
        mytorch.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        ### Update Model ###
        optimizer.step()
        optimizer.zero_grad()

        ### Update Scheduler ###
        scheduler.step()

        return logits, targets, loss

    def generate_sample(starting_text=SAMPLE_GEN_SEED, 
                        gen_len=GEN_LENGTH):
        
        generated = tokenizer.encode(starting_text)

        with mytorch.no_grad():
            for _ in range(gen_len):

                # Limit context to last seq_len tokens
                context = generated[-CONTEXT_LENGTH:]
                context_tensor = mytorch.Tensor(cp.array([context], dtype=cp.int32))

                # Current context length
                curr_len = context_tensor.shape[1]

                # Create causal mask (upper-triangular with -inf)
                causal_mask_data = mytorch.Tensor(
                    cp.triu(cp.ones((curr_len, curr_len), dtype=cp.float32) * float('-inf'), k=1)
                )

                # Forward pass
                logits = model(context_tensor, causal_mask_data)
                last_logits = logits.data[0, -1]  # logits for last position

                # Softmax + sample
                exp_logits = cp.exp(last_logits - cp.max(last_logits))
                probs = (exp_logits / cp.sum(exp_logits))# move to CPU
                next_token = cp.random.choice(tokenizer.vocab_size, p=probs, size=(1,)).get()[0]

                # Append generated token
                generated.append(next_token)
        
        generated = tokenizer.decode(generated)
        
        print("Generated Sample")
        print(generated)
        print("-----------------")

        return generated

    for iter in tqdm(range(TRAINING_ITERATIONS)):

        logits, targets, loss = train_step()

        if iter % LOG_ITER == 0:
            preds = logits.argmax(dim=-1).reshape(-1)
            targets = targets.reshape(-1)
            
            accuracy = (preds == targets).sum() / len(targets) * 100
            print(f"Iteration {iter}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2f}%, LR: {scheduler.get_last_lr():.6f}")

            if LOG_WANDB:
                wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()}, step=iter)

        if iter % GEN_ITER == 0:

            model.eval()
            generate_sample()
            model.train()
        
    ### Save Model ###
    print(f"Saving Weights to {SAVE_PATH}")
    mytorch.save(model.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tiny GPT2 with MyTorch")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--use_gpt2_tokenizer", action="store_true")

    parser.add_argument("--train_iterations", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument("--log_iter", type=int, default=100)
    parser.add_argument("--gen_iter", type=int, default=500)
    parser.add_argument("--sample_seed", type=str, default="h")
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--save_path", type=str, default="model.safetensors")

    args = parser.parse_args()
    main(args)