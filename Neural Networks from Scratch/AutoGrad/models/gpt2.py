import cupy as cp
import mytorch.nn as nn

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
        avail_idx = cp.arange(0, seq_length, dtype=cp.int32)
        pos_embed = self.position_embeddings(avail_idx).reshape(1, seq_length, self.embed_dim)
        x = x + pos_embed

        return x
    
class Attention(nn.Module):

    def __init__(self, embed_dim, num_heads, attn_dropout_p=0.1):
        super().__init__()
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

        ### Compute Projections and make Contiguous ###
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()
        
        ### Compute Attention (Attention Mask has shape Batch x Sequence len x Sequence len) ###
        scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
    
        ### Add attention mask if it exists ###
        if attention_mask is not None:
            scores += attention_mask
     
        attention = self.softmax(scores, dim=-1)
        attention = self.attn_drop(attention)

        output = attention @ v
        output = output.transpose(1,2).reshape(batch,seq_len,embed_dim)
        
        ### Compute Output Projection (on flattened dimension) ###
        output = self.out_proj(output)
        output = self.proj_drop(output)

        return output
    
class FeedForward(nn.Module):
    """
    Regular MLP module after our attention computation. 
    """
    def __init__(self, embed_dim, mlp_ratio=4, mlp_dropout_p=0.1):
        super().__init__()
        hidden_size = embed_dim * mlp_ratio
        self.intermediate_dense = nn.Linear(embed_dim, hidden_size)
        self.activation = nn.GELU()
        self.intermediate_dropout = nn.Dropout(mlp_dropout_p)

        self.output_dense = nn.Linear(hidden_size, embed_dim)
        self.output_dropout = nn.Dropout(mlp_dropout_p)

    def forward(self, x):

        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)

        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p, mlp_ratio=4):
        super().__init__()
        self.attention = Attention(embed_dim, num_heads, dropout_p)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim, mlp_ratio, dropout_p)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attention_mask=None):

        x = self.layernorm1(x)
        x = x + self.attention(x, attention_mask)

        x = self.layernorm2(x)
        x = x + self.feedforward(x)
        
        return x

class GPT2(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 max_seq_len, 
                 embed_dim, 
                 num_heads, 
                 num_blocks, 
                 dropout_p=0.0, 
                 mlp_ratio=4):
        super().__init__()
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
        self.lm_head.W.data = self.embeddings.char_embeddings.weight.data.T

    def forward(self, x, attention_mask=None):

        x = self.embeddings(x)

        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.lm_head(x)

        return x