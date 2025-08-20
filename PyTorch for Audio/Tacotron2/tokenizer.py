import torch

class Tokenizer:

    def __init__(self):
        
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        self.chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? ') + \
            [self.eos_token, self.pad_token, self.unk_token]
        
        self.char2id = {char: i for i, char in enumerate(self.chars)}
        self.id2char = {i: char for i, char in enumerate(self.chars)}

        self.eos_token_id = self.char2id[self.eos_token]
        self.pad_token_id = self.char2id[self.pad_token]
        self.unk_token_id = self.char2id[self.unk_token]
        
    def encode(self, text, return_tensor=True):

        ### Grab Tokens from Text, Use UNK token for anything not known, and end with EOS Token ###
        tokens = [self.char2id.get(char, self.unk_token_id) for char in text] + [self.eos_token_id]

        if return_tensor:
            tokens = torch.tensor(tokens, dtype=torch.long)

        return tokens

    def decode(self, token_ids, include_special_tokens=False):
        
        # Convert IDs to characters, filtering out special tokens if requested
        chars = []
        for token_id in token_ids:
            char = self.id2char.get(token_id, self.unk_token)
            if include_special_tokens or char not in [self.eos_token, self.pad_token, self.unk_token]:
                chars.append(char)
                
        return ''.join(chars)