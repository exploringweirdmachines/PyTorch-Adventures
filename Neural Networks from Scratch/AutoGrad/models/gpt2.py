import numpy as np
import mytorch
import mytorch.nn as nn
import mytorch.nn.functional as F

class Embeddings(nn.Module):

    def __init__(self, vocab_size, embed_dim, context_length):
        super().__init__()
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
        avail_idx = mytorch.arange(start=0, end=seq_length).to(input_ids.device)
        pos_embed = self.position_embeddings(avail_idx).reshape(1, seq_length, self.embed_dim)
        x = x + pos_embed

        return x

class Attention(nn.Module):

    def __init__(self, embed_dim, num_heads, attn_dropout_p=0.1, use_bias=True, fused=False):
        super().__init__()
        ### Sanity Checks ###
        assert embed_dim % num_heads == 0, "Double check embedding dim divisible by number of heads"

        ### Attention Head Dim ###
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.fused = fused

        # ### Attention Projections ###
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.softmax = nn.Softmax()
        self.attn_drop = nn.Dropout(dropout_p=attn_dropout_p)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.proj_drop = nn.Dropout(dropout_p=attn_dropout_p)
        

    def forward(self, x, attention_mask=None):
        batch, seq_len, embed_dim = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*embed_dim]

        # Reshape to multi-head
        qkv = qkv.reshape(batch, seq_len, self.num_heads, 3 * self.head_dim)

        # Transpose to [batch, num_heads, seq_len, 3*head_dim]
        qkv = qkv.transpose(1, 2)

        # Chunk last dim into q, k, v
        q, k, v = mytorch.chunk(qkv, 3, dim=-1)  # each [batch, num_heads, seq_len, head_dim]

        if not self.fused:
        # Compute attention scores
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

            if attention_mask is not None:
                scores = scores + attention_mask.astype(scores.data.dtype)

            attention = self.softmax(scores, dim=-1)
            attention = self.attn_drop(attention)

            # Attention output
            output = attention @ v

        else:
            output = F.scaled_dot_product_attention(q, k, v, causal=True)
            
        output = output.transpose(1, 2).reshape(batch, seq_len, embed_dim)
        # Output projection
        output = self.out_proj(output)
        output = self.proj_drop(output)

        return output
    
class FeedForward(nn.Module):
    """
    Regular MLP module after our attention computation. 
    """
    def __init__(self, 
                 embed_dim, 
                 mlp_ratio=4, 
                 mlp_dropout_p=0.1,
                 use_bias=True):
        super().__init__()
        hidden_size = embed_dim * mlp_ratio

        self.intermediate_dense = nn.Linear(embed_dim, hidden_size, bias=use_bias)
        self.activation = nn.GELU()
        self.intermediate_dropout = nn.Dropout(mlp_dropout_p)

        self.out_proj = nn.Linear(hidden_size, embed_dim, bias=use_bias)
        self.output_dropout = nn.Dropout(mlp_dropout_p)

    def forward(self, x):

        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)

        x = self.out_proj(x)
        x = self.output_dropout(x)

        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 dropout_p, 
                 mlp_ratio=4,
                 use_bias=True):
        super().__init__()
        self.attention = Attention(embed_dim, num_heads, dropout_p, use_bias)
        self.layernorm1 = nn.LayerNorm(embed_dim, bias=use_bias)
        self.feedforward = FeedForward(embed_dim, mlp_ratio, dropout_p, use_bias)
        self.layernorm2 = nn.LayerNorm(embed_dim, bias=use_bias)

    def forward(self, x, attention_mask=None):
        
        attn_out = self.attention(self.layernorm1(x), attention_mask)
        x = x + attn_out
        mlp_out = self.feedforward(self.layernorm2(x))
        x = x + mlp_out
     
        return x

class GPT2(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 max_seq_len, 
                 embed_dim, 
                 num_heads, 
                 num_blocks, 
                 dropout_p=0.0, 
                 mlp_ratio=4,
                 use_bias=True):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.embeddings = Embeddings(vocab_size=vocab_size, 
                                     embed_dim=embed_dim, 
                                     context_length=max_seq_len)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, 
                             num_heads=num_heads, 
                             dropout_p=dropout_p, 
                             mlp_ratio=mlp_ratio,
                             use_bias=use_bias)

            for _ in range(num_blocks)
        ])

        self.final_layer_norm = nn.LayerNorm(embed_dim=embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=True)

        ### Initialize Weights ###
        self.apply(_init_weights)
        for name, param in self.named_parameters():
            if "out_proj" in name:
                mytorch.nn.init.normal_(param, mean=0.0, std=(0.02/np.sqrt(2 * num_blocks)))

        self.lm_head.weight.data = self.embeddings.char_embeddings.weight.data


    def forward(self, x, attention_mask=None):

        x = self.embeddings(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.final_layer_norm(x)    
        x = self.lm_head(x)

        return x
    
### Standard Weight Init for Transformers ###
def _init_weights(module):
    if isinstance(module, nn.Linear):
        mytorch.nn.init.normal_(module.weight, mean=0, std=0.02)
        if module.bias is not None:
            mytorch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        mytorch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, nn.LayerNorm):
        mytorch.nn.init.ones_(module.weight)
        if module.bias is not None:
            mytorch.nn.init.zeros_(module.bias)