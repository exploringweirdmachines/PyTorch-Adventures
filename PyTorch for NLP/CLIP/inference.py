from datasets import load_dataset
from clip import CLIP
import torch
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

### PROMPT ###
prompt = "A person surfing at the beach"

### Load Processor/Tokenizer ###
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

## Load Test Split of Dataset ###
dataset = load_dataset("nlphuji/flickr30k")["test"]
split = dataset.train_test_split(test_size=0.05, seed=42)
test_dataset = split["test"]

### Load Model ###
model = CLIP().to("cuda")
weights = load_file("work_dir/clip_flicker30k/final_checkpoint/model.safetensors")
model.load_state_dict(weights)
model.eval()

### Vectorize all test images ###
image_embeddings = []
counter = 0
for sample in tqdm(test_dataset):
    pixel_values = image_processor(sample["image"], return_tensors="pt")["pixel_values"][0].unsqueeze(0)
    
    with torch.no_grad():
        image_embedding = model.compute_image_embeds(pixel_values.to("cuda"))
    image_embeddings.append(image_embedding)
    counter += 1

image_embeddings = torch.concatenate(image_embeddings)

### Tokenize and Encode Prompt ###
text_inputs = tokenizer(prompt, padding=True, truncation=True, max_length=77, return_tensors="pt")

with torch.no_grad():
    text_embeddings = model.compute_text_embeds(input_ids=text_inputs["input_ids"].to("cuda"),
                                                attention_mask=text_inputs["attention_mask"].to("cuda"))

### Compute Cosine Sim ###
cos_sim = (image_embeddings @ text_embeddings.T).squeeze(-1)

# Get top-k results
k = 5
topk_vals, topk_indices = torch.topk(cos_sim, k=k)

print("Top-k similarities:", topk_vals.tolist())

# Plot top-k images
fig, axes = plt.subplots(1, k, figsize=(4*k, 6)) 
for i, idx in enumerate(topk_indices):
    img = test_dataset[int(idx)]["image"]  # dataset returns PIL image
    axes[i].imshow(img)
    axes[i].set_title(f"Rank {i+1}\nScore: {topk_vals[i]:.2f}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()