import os
import argparse
import numpy as np
import cupy as cp
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from models.gpt2 import GPT2
from miniddp.accelerate import Accelerator
from tqdm import tqdm

GPT2_VOCAB_SIZE = 50257

parser = argparse.ArgumentParser(description="Train Tiny GPT2 with MyTorch")
parser.add_argument("--context_length", type=int, default=512)
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
parser.add_argument("--warmup_steps", type=int, default=0)
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
def get_batch(train=True):

    if train:
        data = np.memmap(os.path.join(args.data_path, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(args.data_path, 'test.bin'), dtype=np.uint16, mode='r')

    start_idx = np.random.randint(low=0, high=len(data) - args.context_length - 1, size=(args.batch_size//args.gradient_accumulation_steps))
    x = np.stack([data[i:i+args.context_length] for i in start_idx])
    y = np.stack([data[i+1:i+args.context_length+1] for i in start_idx])

    return mytorch.Tensor(x), mytorch.Tensor(y)

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

### Load Model ###
model = GPT2(vocab_size=GPT2_VOCAB_SIZE, 
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
    inputs, targets = get_batch(train=True)
    
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

    ### Update Scheduler ###
    scheduler.step()

    if iter % args.gradient_accumulation_steps == 0:
        completed_steps += 1
        pbar.update(1)

        # Gather metrics across GPUs
        if completed_steps % args.log_iter == 0:
            loss_val = accelerator.gather_for_metrics(loss)

            accuracy_val = accelerator.gather_for_metrics(
                (logits.argmax(dim=-1).reshape(-1) == targets.reshape(-1)).sum() / len(targets.reshape(-1))
            )

            accelerator.print(f"Iter {completed_steps}, Loss: {loss_val:.4f}, LR: {scheduler.get_last_lr():.2e}, Acc: {accuracy_val:.4f}")

            if args.log_wandb:
                accelerator.log({"loss": loss_val, "lr": scheduler.get_last_lr()}, step=iter)
                
        
        
### Save Model ###
print(f"Saving Weights to {args.save_path}")
mytorch.save(model.state_dict(), args.save_path)

# End training
accelerator.end_training()  
        
    
