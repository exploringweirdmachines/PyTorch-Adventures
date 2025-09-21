import mytorch
import cupy as cp
from tiny_gpt2 import GPT2
from transformers import GPT2TokenizerFast
from tqdm import tqdm

tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
context_length = 256
model = GPT2(vocab_size=tokenizer.vocab_size, 
             max_seq_len=context_length, 
             embed_dim=384, 
             num_heads=6, 
             num_blocks=6,  
             mlp_ratio=4)

model.load_state_dict(mytorch.load(filepath="work_dir/mini_gpt2_harry_potter.safetensors"))


def generate_sample(starting_text, gen_len):
    
    generated = tokenizer.encode(starting_text)

    with mytorch.no_grad():
        for _ in tqdm(range(gen_len)):

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

    return generated

if __name__ == "__main__":

    generated = generate_sample("The spell was cast", gen_len=512)
    print(generated)
