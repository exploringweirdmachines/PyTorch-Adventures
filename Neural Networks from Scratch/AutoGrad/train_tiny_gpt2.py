import os
import cupy as cp
from tqdm import tqdm
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from tiny_gpt2 import GPT2
from transformers import GPT2TokenizerFast
import wandb

### MODEL PARAMETERS ###
CONTEXT_LENGTH = 256
EMBED_DIM = 384
NUM_HEADS = 6
NUM_BLOCKS = 6
DROPOUT_P = 0.1
MLP_RATIO = 4
PATH_TO_DATA = "../../data/harry_potter_txt"

### TRAINING PARAMETERS ###
TRAINING_ITERATIONS = 25000
BATCH_SIZE = 64
MAX_LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 5e-5
WARMUP_STEPS = 5000
WEIGHT_DECAY = 0.05
MAX_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 1

### LOGGING PARAMETERS ###
LOG_ITER = 25
GEN_ITER = 1000
SAMPLE_GEN_SEED = "Spells "
GEN_LENGTH = 256

### SAVE WEIGHTS PARAMETERS ###
SAVE_PATH =  "work_dir/mini_gpt2_harry_potter.safetensors"

### INITIALIZE WANDB ###
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
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokens = cp.array(tokenizer.encode(all_text))

### Grab Batches from Dataset ###
def get_batch(data, batch_size, seq_len):
    idx = cp.random.randint(0, len(data) - seq_len - 1, batch_size)
    inputs = cp.stack([data[i:i+seq_len] for i in idx])
    targets = cp.stack([data[i+1:i+seq_len+1] for i in idx])
    return inputs, targets.flatten()

### Load Model ###
model = GPT2(vocab_size=tokenizer.vocab_size, 
             max_seq_len=CONTEXT_LENGTH, 
             embed_dim=EMBED_DIM, 
             num_heads=NUM_HEADS, 
             num_blocks=NUM_BLOCKS, 
             dropout_p=DROPOUT_P, 
             mlp_ratio=MLP_RATIO)

### Load Causal Mask for Training ###
causal_mask = mytorch.Tensor(
    cp.triu(cp.ones((1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH)) * float('-inf'), k=1)
).astype(cp.float32)

### Load Optimizer ###
optimizer = optim.AdamW(model.parameters(), lr=MAX_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

### Load Scheduler ###
scheduler = mytorch.utils.CosineLRScheduler(
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
        accuracy = (logits.argmax(dim=-1).reshape(-1) == targets).sum() / len(targets) * 100
        print(f"Iteration {iter}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2f}%, LR: {scheduler.get_last_lr():.6f}")

        wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()}, step=iter)

    if iter % GEN_ITER == 0:

        model.eval()

        generated_text = generate_sample()
        wandb.log({"sample_text": generated_text}, step=iter)

        model.train()

### Save Model ###
print(f"Saving Weights to {SAVE_PATH}")
mytorch.save(model.state_dict(), SAVE_PATH)

# # 5. Training loop

# for epoch in tqdm(range(TRAINING_ITERATIONS)):
#     inputs, targets = get_batch(data, BATCH_SIZE, seq_len)
#     logits = model(inputs, causal_mask)
#     loss = loss_fn(logits, targets)
#     loss.backward()
    
#     optimizer.step()
#     optimizer.zero_grad()

#     if epoch % 100 == 0:
#         accuracy = (logits.argmax(dim=-1).reshape(-1) == targets).sum() / len(targets) * 100
#         print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2f}%")


# # mytorch.save(model.state_dict(), "work_dir/character_transformer_small.safetensors")
# state_dict = mytorch.load("work_dir/character_transformer_small.safetensors")
# model.load_state_dict(state_dict)

# # # Evaluation 
# model.eval()
# gen_len = 1024

# # # Initialize seed tensor (make sure dtype is long/int for indices)
# seed = mytorch.Tensor(cp.array([[char_to_idx['h']]])) 
# generated = list(seed.data.get()[0])  # start with initial seed

# with mytorch.no_grad():
#     for _ in range(gen_len):

#         # Limit context to last seq_len tokens
#         context = generated[-CONTEXT_LENGTH:]
#         context_tensor = mytorch.Tensor(cp.array([context], dtype=cp.int32))

#         # Current context length
#         curr_len = context_tensor.shape[1]

#         # Create causal mask (upper-triangular with -inf)
#         causal_mask_data = mytorch.Tensor(
#             cp.triu(cp.ones((curr_len, curr_len), dtype=cp.float32) * float('-inf'), k=1)
#         )

#         # Forward pass
#         logits = model(context_tensor, causal_mask_data)
#         last_logits = logits.data[0, -1]  # logits for last position

#         # Softmax + sample
#         exp_logits = cp.exp(last_logits - cp.max(last_logits))
#         probs = (exp_logits / cp.sum(exp_logits)).get()  # move to CPU
#         next_token = np.random.choice(vocab_size, p=probs)

#         # Append generated token
#         generated.append(next_token)

# # Convert indices back to characters
# print("".join([idx_to_char[i] for i in generated]))