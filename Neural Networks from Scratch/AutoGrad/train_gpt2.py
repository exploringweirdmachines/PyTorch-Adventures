import os
import argparse
import numpy as np
import cupy as cp
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from models.gpt2 import GPT2, GPT2Config
from miniddp.accelerate import Accelerator
from tqdm import tqdm

GPT2_VOCAB_SIZE = 50257

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT2 with MyTorch")

    ### Experiment Config ###
    parser.add_argument("--project_name", type=str, default="GPT2Trainer")
    parser.add_argument("--run_name", type=str)

    ### Checkpointing Config ###
    parser.add_argument("--working_directory", type=str, required=True)
    parser.add_argument("--checkpoint_iter", type=int, default=10000)
    parser.add_argument("--resume_from_checkpoint", type=str)

    ### Model Config ###
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_blocks", type=int, default=12)
    parser.add_argument("--dropout_p", type=float, default=0.)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--fused", action="store_true")

    ### Training Config ###
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--train_iterations", type=int, default=150000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--mixed_precision", action="store_true")

    ### Logging Config ###
    parser.add_argument("--log_iter", type=int, default=100)
    parser.add_argument("--log_wandb", action="store_true")
    

    args = parser.parse_args()

    return args

### Load Arguments ###
args = parse_args()

### LOAD DATASET ###
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
path_to_experiment = os.path.join(args.working_directory, args.project_name)
if args.run_name is not None:
    path_to_experiment = os.path.join(path_to_experiment, args.run_name)

accelerator = Accelerator(project_dir=path_to_experiment, 
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          mixed_precision=args.mixed_precision)

if args.log_wandb:
    training_config = {
                        "context_length": args.context_length,
                        "embed_dim": args.embed_dim,
                        "num_heads": args.num_heads,
                        "num_blocks": args.num_blocks,
                        "dropout": args.dropout_p,
                        "batch_size": args.batch_size,
                        "grad_accumulation_steps": args.gradient_accumulation_steps,
                        "max_lr": args.max_lr,
                        "weight_decay": args.weight_decay,
                        "adam_beta1": args.beta1,
                        "adam_beta2": args.beta2,
                        "mixed_precision": args.mixed_precision
                    }

    accelerator.init_tracker(project_name=args.project_name, 
                             run_name=args.run_name,
                             config=training_config)

### Load Model ###
config = GPT2Config(
        vocab_size=GPT2_VOCAB_SIZE,
        max_seq_len=args.context_length,
        embed_dim=args.embed_dim, 
        num_heads=args.num_heads,
        num_blocks=args.num_blocks, 
        attn_dropout_p=args.dropout_p, 
        mlp_dropout_p=args.dropout_p, 
        mlp_ratio=args.mlp_ratio,
        use_bias=args.use_bias, 
        use_fused_ops=args.fused
    )
model = GPT2(config)

total_params = 0
for param in model.parameters():
    if param.requires_grad:
        total_params += np.prod(param.shape)
accelerator.print("Total Trainable Parameters:", total_params)

### Load Optimizer ###
optimizer = optim.AdamW(model.parameters(), 
                        lr=args.max_lr, 
                        weight_decay=args.weight_decay,
                        beta1=args.beta1, 
                        beta2=args.beta2)

### Load Scheduler ###
scheduler = mytorch.lr_scheduler.CosineLRScheduler(
    optimizer=optimizer, max_lr=args.max_lr, 
    min_lr=args.min_lr, total_steps=args.train_iterations,
    warmup_steps=args.warmup_steps
)

### Prepare Everything ###
model, optimizer = accelerator.prepare(model, optimizer)

### Resume from Checkpoint ###
if args.resume_from_checkpoint is not None:

    ### Grab path to checkpoint ###
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    
    ### Load our State (model and optimizer) ###
    accelerator.load_state(path_to_checkpoint)
    
    ### Start completed steps from checkpoint index ###
    completed_steps = int(args.resume_from_checkpoint.split("_")[-1])
    accelerator.print(f"Resuming from Iteration: {completed_steps}")

    ### Advance our scheduler to the correct step ###
    scheduler.step_count = completed_steps

else:
    completed_steps = 0

### Load Loss Function ###
loss_fn = nn.CrossEntropyLoss()

### Train Model ###
pbar = tqdm(range(args.train_iterations), 
            disable=not accelerator.is_main_process(),
            initial=completed_steps)

for iter in range((args.train_iterations - completed_steps) * args.gradient_accumulation_steps):

    # Sample a batch
    inputs, targets = get_batch(train=True)
    inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)

    # Forward pass
    logits = model(inputs)
    loss = loss_fn(logits, targets)

    # Backward
    accelerator.backward(loss)
    
    # Clip gradients (and get the grads to check on training health)
    grad_norm = accelerator.clip_grad_norm_(args.max_grad_norm)

    # Step optimizer
    optimizer.step()
    optimizer.zero_grad()

    ### Accelerator tracks when accumulation is done, the flag is just sync_grad ###
    if accelerator.sync_grad:

        completed_steps += 1
        pbar.update(1)

        ### Update Scheduler ###
        scheduler.step()

        # Gather metrics across GPUs
        if completed_steps % args.log_iter == 0:
            loss_val = accelerator.gather_for_metrics(loss)

            accuracy_val = accelerator.gather_for_metrics(
                (logits.argmax(dim=-1).reshape(-1) == targets.reshape(-1)).sum() / len(targets.reshape(-1))
            )

            ### Grab our stored grad_norm for checking on model health ###
            grad_norm = accelerator.grad_norm
            log_statement = f"Iter {completed_steps}, Loss: {loss_val:.4f}, LR: {scheduler.get_last_lr():.2e}, Acc: {accuracy_val:.4f}"
            if grad_norm is not None:
                log_statement += f" Grad Norm: {grad_norm:.3f}"

            accelerator.print(log_statement)

            if args.log_wandb:
                accelerator.log({"loss": loss_val, "lr": scheduler.get_last_lr(), "grad_norm": grad_norm}, step=completed_steps)

            if completed_steps % args.checkpoint_iter == 0 :
                accelerator.save_state(os.path.join(path_to_experiment, f"checkpoint_{completed_steps}"))


# ### Save Model ###
# accelerator.save_state(os.path.join(path_to_experiment, f"final_checkpoint"))

# # End training
# accelerator.end_training()  
        
    
