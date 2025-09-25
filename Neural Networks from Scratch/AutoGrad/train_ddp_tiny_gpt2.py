import os
import argparse
import numpy as np
import cupy as cp
import requests
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from models.gpt2 import GPT2
from miniddp.accelerate import Accelerator
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train Tiny GPT2 with MyTorch")
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--embed_dim", type=int, default=384)
parser.add_argument("--num_heads", type=int, default=6)
parser.add_argument("--num_blocks", type=int, default=6)
parser.add_argument("--dropout_p", type=float, default=0.1)
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
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

parser.add_argument("--log_iter", type=int, default=100)
parser.add_argument("--gen_iter", type=int, default=500)
parser.add_argument("--sample_seed", type=str, default="h")
parser.add_argument("--gen_length", type=int, default=256)
parser.add_argument("--log_wandb", action="store_true")
parser.add_argument("--project_name", type=str, default="MyTorch_Char_GPT2")
parser.add_argument("--run_name", type=str)
parser.add_argument("--save_path", type=str, default="model.safetensors")

args = parser.parse_args()

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
        
    return all_txt

def generate_sample(model, tokenizer, starting_text, gen_len, context_length):
        
        generated = tokenizer.encode(starting_text)

        with mytorch.no_grad():
            for _ in range(gen_len):

                # Limit context to last seq_len tokens
                context = generated[-context_length:]
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

### INIT ACCELERATOR ###
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

if args.log_wandb:
    training_config = {
                        "context_length": args.context_length,
                        "embed_dim": args.embed_dim,
                        "num_heads": args.num_heads,
                        "num_blocks": args.num_blocks,
                        "dropout": args.dropout_p,
                        "batch_size": args.batch_size,
                        "learning_rate": args.max_lr,
                        "weight_decay": args.weight_decay
                    }

    accelerator.init_tracker(project_name=args.project_name, 
                            run_name=args.run_name,
                            config=training_config)

### Load Text ###s
all_text = get_text_data(args.data_path)

### Character Tokenize Text ###
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
    return mytorch.Tensor(inputs), mytorch.Tensor(targets.flatten())

### Load Model ###
model = GPT2(vocab_size=tokenizer.vocab_size, 
             max_seq_len=args.context_length, 
             embed_dim=args.embed_dim, 
             num_heads=args.num_heads, 
             num_blocks=args.num_blocks, 
             dropout_p=args.dropout_p, 
             mlp_ratio=args.mlp_ratio)

accelerator.print(model)
total_params = 0
for param in model.parameters():
    if param.requires_grad:
        total_params += np.prod(param.shape)
accelerator.print("Total Trainable Parameters:", total_params)

### Load Causal Mask for Training ###
causal_mask = mytorch.Tensor(
    cp.triu(cp.ones((1, 1, args.context_length, args.context_length)) * float('-inf'), k=1)
).astype(cp.float32)

### Load Optimizer ###
optimizer = optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

### Load Scheduler ###
scheduler = mytorch.lr_scheduler.CosineLRScheduler(
    optimizer=optimizer, max_lr=args.max_lr, 
    min_lr=args.min_lr, total_steps=args.train_iterations,
    warmup_steps=args.warmup_steps
)

### Prepare Everything ###
model, optimizer = accelerator.prepare(model, optimizer)

### Load Loss Function ###
loss_fn = nn.CrossEntropyLoss()

### Train Model ###

pbar = tqdm(range(args.train_iterations), disable=not accelerator.is_main_process())
completed_steps = 0
for iter in range(args.train_iterations * args.gradient_accumulation_steps):

    # Sample a batch
    inputs, targets = get_batch(tokens, args.batch_size // args.gradient_accumulation_steps, args.context_length)
    
    # Forward pass
    logits = model(inputs, causal_mask)
    loss = loss_fn(logits, targets)

    # Backward
    accelerator.backward(loss)

    # Clip gradients
    accelerator.clip_grad_norm_(args.max_grad_norm)

    # Step optimizer
    optimizer.step()
    optimizer.zero_grad()

    if iter % args.gradient_accumulation_steps == 0:
        
        ### Update Scheduler ###
        scheduler.step()
        
        ### Update Progress Bar ###
        pbar.update(1)
        completed_steps += 1
    
        if completed_steps % args.log_iter == 0:
            loss_val = accelerator.gather_for_metrics(loss)
            accuracy_val = accelerator.gather_for_metrics(
                (logits.argmax(dim=-1).reshape(-1) == targets).sum() / len(targets)
            )
            accelerator.print(f"Iter {completed_steps}, Loss: {loss_val:.4f}, Accuracy: {accuracy_val*100:.2f}%, LR: {scheduler.get_last_lr():.2e}")

            if args.log_wandb:
                accelerator.log({"loss": loss_val, "accuracy": accuracy_val, "lr": scheduler.get_last_lr()}, step=completed_steps)

        # Generate sample text every GEN_ITER
        if completed_steps % args.gen_iter == 0 and accelerator.is_main_process():
            model.eval()
            generate_sample(model=model,
                            tokenizer=tokenizer,
                            starting_text=args.sample_seed,
                            gen_len=args.gen_length,
                            context_length=args.context_length)
            model.train()
    
    
        
### Save Model ###
print(f"Saving Weights to {args.save_path}")
mytorch.save(model.state_dict(), args.save_path)

# End training
accelerator.end_training()  
        
    
