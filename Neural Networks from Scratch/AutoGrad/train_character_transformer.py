import cupy as cp
import numpy as np
from tqdm import tqdm
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
import requests

class Embeddings(nn.Module):

    def __init__(self, vocab_size, embed_dim, context_length):

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_length = context_length

        ### Embeddings for Tokens ###
        self.char_embeddings = nn.Embedding(vocab_size, embed_dim)

        ### Positional Embeddings ###
        self.position_embeddings = nn.Embedding(context_length, embed_dim)

    def forward(self, input_ids):

        batch_size, seq_length = input_ids.shape

        ### Convert Tokens to Embeddings ###
        x = self.char_embeddings(input_ids)

        ### Add Positional Information ###
        avail_idx = cp.arange(0, seq_length, dtype=cp.int32)
        pos_embed = self.position_embeddings(avail_idx).reshape(1, seq_length, self.embed_dim)
        x = x + pos_embed

        return x
    
class Attention(nn.Module):

    def __init__(self, embed_dim, num_heads, attn_dropout_p=0.1):
        
        ### Sanity Checks ###
        assert embed_dim % num_heads == 0, "Double check embedding dim divisible by number of heads"

        ### Attention Head Dim ###
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        ### Attention Projections ###
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax()
        self.attn_drop = nn.Dropout(dropout_p=attn_dropout_p)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout_p=attn_dropout_p)
        

    def forward(self, x, attention_mask=None):
     
        ### Store Shape ###
        batch, seq_len, embed_dim = x.shape

        ### Flatten Batch and Seq Len Dimension ###
        x = x.reshape(batch*seq_len, embed_dim)
   
        ### Compute Attention with Flash Attention ###
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
       
        ### Compute Attention (Attention Mask has shape Batch x Sequence len x Sequence len) ###
        scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
    
        ### Add attention mask if it exists ###
        if attention_mask is not None:
            scores += attention_mask
     
        attention = self.softmax(scores, dim=-1)
        attention = self.attn_drop(attention)

        output = attention @ v
        output = output.transpose(1,2).reshape(batch*seq_len, embed_dim)
        
        ### Compute Output Projection (on flattened dimension) ###
        output = self.out_proj(output)
        output = self.proj_drop(output)

        output = output.reshape(batch, seq_len, embed_dim)
        
        return output
    
class FeedForward(nn.Module):
    """
    Regular MLP module after our attention computation. 
    """
    def __init__(self, embed_dim, mlp_ratio=4, mlp_dropout_p=0.1):
        
        hidden_size = embed_dim * mlp_ratio
        self.intermediate_dense = nn.Linear(embed_dim, hidden_size)
        self.activation = nn.ReLU()
        self.intermediate_dropout = nn.Dropout(mlp_dropout_p)

        self.output_dense = nn.Linear(hidden_size, embed_dim)
        self.output_dropout = nn.Dropout(mlp_dropout_p)

    def forward(self, x):

        ### Reshape X to be (B*S x E)
        batch_size, seq_len, embed_dim = x.shape
        x = x.reshape(batch_size*seq_len, embed_dim)

        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)

        ### Return original shape ####
        x = x.reshape(batch_size, seq_len, embed_dim)

        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p, mlp_ratio=4):

        self.attention = Attention(embed_dim, num_heads, dropout_p)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim, mlp_ratio, dropout_p)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attention_mask=None):

        x = x + self.attention(x, attention_mask)
        x = self.layernorm(x)

        x = x + self.feedforward(x)
        x = self.layernorm2(x)
       
        return x

class Transformer(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 max_seq_len, 
                 embed_dim, 
                 num_heads, 
                 num_blocks, 
                 dropout_p, 
                 mlp_ratio=4):
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embeddings = Embeddings(vocab_size=vocab_size, embed_dim=embed_dim, context_length=max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, 
                             num_heads=num_heads, 
                             dropout_p=dropout_p, 
                             mlp_ratio=mlp_ratio)

            for _ in range(num_blocks)
        ])

        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, attention_mask=None):

        batch_size, seq_len = x.shape

        x = self.embeddings(x)

        for block in self.blocks:
            x = block(x, attention_mask)
       
        ### Flatten for Linear Layer ###        
        x = x.reshape(batch_size*seq_len, self.embed_dim)
        x = self.lm_head(x)
        x = x.reshape(batch_size, seq_len, self.vocab_size)

        return x

USE_AUTO_METHDOS = False
if USE_AUTO_METHDOS:
    nn.Linear = nn.AutoLinear
    nn.LayerNorm = nn.AutoLayerNorm
    nn.ReLU = nn.AutoReLU
    nn.Softmax = nn.AutoSoftmax
    nn.CrossEntropyLoss = nn.AutoCrossEntropyLoss

### Get Dataset ###
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
print(f"Dataset length: {len(text)} characters")

# 2. Create character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
data = cp.array([char_to_idx[ch] for ch in text])

### Load Model ###
model = Transformer(vocab_size=vocab_size, 
                    max_seq_len=256, 
                    embed_dim=384, 
                    num_heads=6, 
                    num_blocks=6, 
                    dropout_p=0.1, 
                    mlp_ratio=4)

# 3. Batch generation
def get_batch(data, batch_size, seq_len):
    idx = cp.random.randint(0, len(data) - seq_len - 1, batch_size)
    inputs = cp.stack([data[i:i+seq_len] for i in idx])
    targets = cp.stack([data[i+1:i+seq_len+1] for i in idx])
    return inputs, targets.flatten()

seq_len = 256
causal_mask = mytorch.Tensor(cp.triu(np.ones((1, 1, seq_len, seq_len)) * float('-inf'), k=1)).astype(cp.float32)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 5. Training loop
train_iterations = 500
batch_size = 32

for epoch in tqdm(range(train_iterations)):
    inputs, targets = get_batch(data, batch_size, seq_len)
    logits = model(inputs, causal_mask)
    loss = loss_fn(logits, targets)
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        accuracy = (logits.argmax(dim=-1).reshape(-1) == targets).sum() / len(targets) * 100
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2f}%")

# Evaluation 
model.eval()

# Initialize seed tensor (make sure dtype is long/int for indices)
seed = mytorch.Tensor(cp.array([[char_to_idx['h']]])) 
generated = [seed.data.item()]

with mytorch.no_grad():
    for _ in range(seq_len):
        curr_len = seed.shape[1]

        # Create causal mask (upper-triangular with -inf)
        causal_mask_data = mytorch.Tensor(cp.triu(cp.ones((curr_len, curr_len), dtype=cp.float32) * float('-inf'), k=1))

        # Forward pass
        logits = model(seed, causal_mask_data)
        last_logits = logits.data[0, -1]  # last position logits

        # Softmax + sample
        exp_logits = cp.exp(last_logits - cp.max(last_logits))
        probs = (exp_logits / cp.sum(exp_logits)).get() 
        next_token = np.random.choice(vocab_size, p=probs)

        # Append generated token
        generated.append(next_token)

        # Update seed with new token
        seed = mytorch.Tensor(cp.array([generated], dtype=cp.int32))
        print(seed.shape)

# Convert indices back to characters
print("".join([idx_to_char[i] for i in generated]))