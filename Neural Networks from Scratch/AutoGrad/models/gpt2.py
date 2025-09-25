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

    def __init__(self, embed_dim, num_heads, attn_dropout_p=0.1, use_bias=True):
        super().__init__()
        ### Sanity Checks ###
        assert embed_dim % num_heads == 0, "Double check embedding dim divisible by number of heads"

        ### Attention Head Dim ###
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        ### Attention Projections ###
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.softmax = nn.Softmax()
        self.attn_drop = nn.Dropout(dropout_p=attn_dropout_p)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.proj_drop = nn.Dropout(dropout_p=attn_dropout_p)
        

    def forward(self, x, attention_mask=None):
     
        ### Store Shape ###
        batch, seq_len, embed_dim = x.shape

        ### Compute Projections and make Contiguous ###
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()
        
        ### Compute Attention (Attention Mask has shape Batch x Sequence len x Sequence len) ###
        scores = (q @ k.transpose(-2, -1).contiguous()) / self.head_dim**0.5
    
        ### Add attention mask if it exists ###
        if attention_mask is not None:
            scores += attention_mask
     
        attention = self.softmax(scores, dim=-1)
        attention = self.attn_drop(attention)

        output = attention @ v
        output = output.transpose(1,2).reshape(batch,seq_len,embed_dim).contiguous()
        
        ### Compute Output Projection (on flattened dimension) ###
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
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=use_bias)

        ### Initialize Weights ###
        self.apply(_init_weights)
        for name, param in self.named_parameters():
            if "out_proj" in name:
                param.data = cp.random.normal(
                    loc=0.0, scale=(0.02 / cp.sqrt(2 * num_blocks)), 
                    size=param.shape, dtype=cp.float32
                )
                
        ### Weight Tying ###
        self.lm_head.weight.data = self.embeddings.char_embeddings.weight.data.T

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
        # Normal init for weights
        module.weight.data[...] = cp.random.normal(
            0.0, 0.02, size=module.weight.data.shape
        ).astype(cp.float32)
        print("its happening")

        if module.bias is not None:
            module.bias.data[...] = 0.0  

    elif isinstance(module, nn.Embedding):
        module.weight.data[...] = cp.random.normal(
            0.0, 0.02, size=module.weight.data.shape
        ).astype(cp.float32)

    elif isinstance(module, nn.LayerNorm):
        module.weight.data[...] = 1.0
        if module.bias is not None:
            module.bias.data[...] = 0.0