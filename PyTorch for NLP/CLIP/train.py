import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer
from datasets import load_dataset
import random
import math
from tqdm import tqdm
import torch.distributed as dist
from torchvision import transforms

from clip import CLIP

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Training Script")
    parser.add_argument("--working_directory", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=50000, help="Maximum training steps")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--val_steps", type=int, default=5000, help="Validate every N steps")
    parser.add_argument("--log_steps", type=int, default=5, help="Log every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume from checkpoint")
    parser.add_argument("--log_wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()
    return args

args = parse_args()

### Load Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment, 
                          log_with="wandb" if args.log_wandb else None)
if args.log_wandb:
    accelerator.init_trackers(args.experiment_name)


### Load Dataset ###
dataset = load_dataset("nlphuji/flickr30k")["test"]
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

### Prepare DataLoader ###
def CLIPDataCollator(image_processor, tokenizer):

    # Transform pipeline matching ConvNextImageProcessor + random crop
    rand_crop_transform = transforms.Compose([
        transforms.Resize(image_processor.size["shortest_edge"]),
        transforms.RandomCrop(image_processor.size["shortest_edge"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, 
                             std=image_processor.image_std)
    ])
    
    def collate_fn(batch):

        # Process images with random crop
        images = [rand_crop_transform(item['image'].convert("RGB")) for item in batch]

        # Stack into a single tensor
        pixel_values = torch.stack(images)

        # Sample one caption per image
        captions = [random.choice(item['caption']) for item in batch]

        # Tokenize captions
        text_inputs = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"]
        }
    
    return collate_fn

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=128, 
    shuffle=True, 
    collate_fn=CLIPDataCollator(image_processor, tokenizer),
    num_workers=24,
    drop_last=True
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=128, 
    shuffle=False, 
    collate_fn=CLIPDataCollator(image_processor, tokenizer),
    num_workers=24,
    drop_last=True
)

# Model and optimizer
model = CLIP()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# Prepare for distributed training
model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)
num_processes = accelerator.num_processes  # Define for loss scaling

def custom_gather_features(features, accelerator):

    """
    We want to compute our contrastive loss across all samples on ALL GPUS, so we need to
    collect them. The issue is that methods like .gather() dont copy gradients, so this
    would break backprop. So instead we do a gather for everything, and then replace 
    the current process_idx with our actual data that has gradients. 

    Our goal is to maximize the likelihood of the correct answer (along the diagonal). So we need
    to ensure that all tensors that contributed to diagonal values (on each GPU) has gradiens from 
    both text and image encodings. All other values just act as negatives for the loss computation.

    In cross entorpy loss, we minimize -log(p_i) for the correct y_i so in our first batch (when collecting)
    across GPUs (lets say we have a global batch size of 8 with 4 samples on each gpu if we had 2 gpus)

    GPU1:
    [11, 12, 13, 14, 15, 16, 17, 18]
    [21, 22, 23, 24, 25, 26, 27, 28]
    [31, 32, 33, 34, 35, 36, 37, 38]
    [41, 42, 43, 44, 45, 46, 47, 48]

    GPU2:
    [51, 52, 53, 54, 55, 56, 57, 58]
    [61, 62, 63, 64, 65, 66, 67, 68]
    [71, 72, 73, 74, 75, 76, 77, 78]
    [81, 82, 83, 84, 85, 86, 87, 88]

    where ij represents sample i from one domain and sample j from other domain. Our model
    will maximize the cases where i==j (when they match)

    So our loss really just cares about:

    in GPU1: 11, 22, 33, 44
    in GPU2: 55, 66, 77, 88

    So we need to ensure that we have gradient information for all samples 1-4 in our first GPU
    for both text and image encodings, and similarly all samples 5-8 in our second gpu for both
    test and image encodings. 

    """
    if accelerator.num_processes == 1:
        return features

    # Create empty list of tensors to gather into ###
    gathered_features = [torch.zeros_like(features) for _ in range(accelerator.num_processes)]
    
    # Copy tensors across gpus into list ###
    dist.all_gather(gathered_features, features)

    # Replace local rank's tensor with original (to preserve gradients)
    gathered_features[accelerator.process_index] = features

    # Concatenate to get global tensor
    global_features = torch.cat(gathered_features, dim=0)

    return global_features

# Training loop
model.train()
step = 0
pbar = tqdm(range(args.max_steps))

train = True
while train:
    for batch in train_dataloader:
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        local_bs = pixel_values.shape[0]  # Dynamic for flexibility
        
        # Forward pass
        image_embeds, text_embeds, logit_scale = model(pixel_values, input_ids, attention_mask)
        
        # All-gather embeddings for global batch
        global_image_embeds = custom_gather_features(image_embeds, accelerator)
        global_text_embeds = custom_gather_features(text_embeds, accelerator)
        
        # Clip logit scale
        logit_scale = torch.clamp(logit_scale, max=math.log(100))
        logit_scale_exp = logit_scale.exp()

        # Compute similarity logits (sharded)
        logits_per_image = logit_scale_exp * image_embeds @ global_text_embeds.T
        logits_per_text = logit_scale_exp * text_embeds @ global_image_embeds.T

        # Labels
        start_idx = accelerator.process_index * local_bs
        labels = torch.arange(start_idx, start_idx + local_bs, device=accelerator.device)
        
        # Symmetric contrastive loss
        loss_img = F.cross_entropy(logits_per_image, labels) / num_processes
        loss_txt = F.cross_entropy(logits_per_text, labels) / num_processes
        loss = (loss_img + loss_txt) / 2
        
        # Backprop
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        step += 1
        pbar.update(1)
        
        # Print every log_steps steps
        if step % args.log_steps == 0 and accelerator.is_main_process:
            tau = accelerator.unwrap_model(model).tau.exp().item()
            print(f"Step {step}, Loss: {loss.item():.4f}, Tau: {tau:.4f}")
            if args.log_wandb:
                accelerator.log({"train_loss": loss.item(), "tau": tau}, step=step)

        # Save checkpoint 
        if step % args.save_steps == 0:
            path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{step}")
            accelerator.wait_for_everyone()
            ### Save checkpoint using only the main process ###
            if accelerator.is_main_process:
                accelerator.save_state(output_dir=path_to_checkpoint)
        
        # Validation every 5000 steps
        if step % args.val_steps == 0:
            val_losses = []
            model.eval()
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_pixel_values = val_batch["pixel_values"]
                    val_input_ids = val_batch["input_ids"]
                    val_attention_mask = val_batch["attention_mask"]
                    
                    val_image_embeds, val_text_embeds, val_logit_scale = model(val_pixel_values, val_input_ids, val_attention_mask)
                    
                    # For validation, use gather_for_metrics (no grads needed)
                    val_global_image_embeds = accelerator.gather_for_metrics(val_image_embeds)
                    val_global_text_embeds = accelerator.gather_for_metrics(val_text_embeds)
                    
                    # Shard similarities in val for consistency (though not strictly needed if val batches are small)
                    val_logit_scale = torch.clamp(val_logit_scale, max=math.log(100))
                    val_logits_per_image = val_logit_scale.exp() * val_image_embeds @ val_global_text_embeds.T
                    val_logits_per_text = val_logit_scale.exp() * val_text_embeds @ val_global_image_embeds.T
                    
                    val_local_bs = len(val_image_embeds)
                    val_start_idx = accelerator.process_index * val_local_bs
                    val_labels = torch.arange(val_start_idx, val_start_idx + val_local_bs, device=accelerator.device)
                    
                    val_loss_img = F.cross_entropy(val_logits_per_image, val_labels) / num_processes
                    val_loss_txt = F.cross_entropy(val_logits_per_text, val_labels) / num_processes
                    val_loss = (val_loss_img + val_loss_txt) / 2
                    
                    # Gather the loss value across processes for accurate averaging
                    val_loss_gathered = accelerator.reduce(val_loss, reduction="mean")
                    val_losses.append(val_loss_gathered.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            accelerator.print(f"Step {step}, Validation Loss: {avg_val_loss:.4f}")
            if args.log_wandb:
                accelerator.log({"val_loss": avg_val_loss}, step=step)
            
            model.train()
        
        if step >= args.max_steps:
            train = False
            accelerator.print("Completed Training")
            break

# Save model
accelerator.save_state(os.path.join(path_to_experiment, "final_checkpoint"))

accelerator.end_training()