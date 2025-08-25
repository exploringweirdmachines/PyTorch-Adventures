import random
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, Sampler
import librosa
from tokenizer import Tokenizer
import matplotlib.pyplot as plt

import numpy as np

def load_wav(path_to_audio, sr=22050):
    audio, orig_sr = torchaudio.load(path_to_audio)

    if sr != orig_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sr)

    return audio.squeeze(0)

def amp_to_db(x):
    ### Forces min DB to be -100
    ### 20 * torch.log10(1e-5) = 20 * -5 = -100
    return 20 * torch.log10(torch.clamp(x, min=1e-5))

def db_to_amp(x):
    return 10 ** (x / 20)

def normalize(x, 
              min_db=-100., 
              max_abs_val=4):

    x = (x - min_db) / -min_db
    x = 2 * max_abs_val * x - max_abs_val
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    
    return x

def denormalize(x, 
                min_db=-100, 
                max_abs_val=4):
    
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    x = (x + max_abs_val) / (2 * max_abs_val)
    x = x * -min_db + min_db

    return x

class AudioMelConversions:
    def __init__(self,
                 num_mels=80,
                 sampling_rate=22050, 
                 n_fft=1024, 
                 window_size=1024, 
                 hop_size=256,
                 fmin=0, 
                 fmax=8000,
                 center=False,
                 min_db=-100, 
                 max_scaled_abs=4):
        
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.spec2mel = self._get_spec2mel_proj()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)

    def _get_spec2mel_proj(self):
        mel = librosa.filters.mel(sr=self.sampling_rate, 
                                  n_fft=self.n_fft, 
                                  n_mels=self.num_mels, 
                                  fmin=self.fmin, 
                                  fmax=self.fmax)
        return torch.from_numpy(mel)
    
    def audio2mel(self, audio, do_norm=False):

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        spectrogram = torch.stft(input=audio, 
                                 n_fft=self.n_fft, 
                                 hop_length=self.hop_size, 
                                 win_length=self.window_size, 
                                 window=torch.hann_window(self.window_size).to(audio.device), 
                                 center=self.center, 
                                 pad_mode="reflect", 
                                 normalized=False, 
                                 onesided=True,
                                 return_complex=True)
        
        spectrogram = torch.abs(spectrogram)
        
        mel = torch.matmul(self.spec2mel.to(spectrogram.device), spectrogram)

        mel = amp_to_db(mel)
        
        if do_norm:
            mel = normalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        return mel
    
    def mel2audio(self, mel, do_denorm=False, griffin_lim_iters=60):

        if do_denorm:
            mel = denormalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        mel = db_to_amp(mel)

        spectrogram = torch.matmul(self.mel2spec.to(mel.device), mel).cpu().numpy()

        audio = librosa.griffinlim(S=spectrogram, 
                                   n_iter=griffin_lim_iters, 
                                   hop_length=self.hop_size, 
                                   win_length=self.window_size, 
                                   n_fft=self.n_fft,
                                   window="hann")

        audio *= 32767 / max(0.01, np.max(np.abs(audio)))
        
        audio = audio.astype(np.int16)

        return audio

def build_padding_mask(lengths):

    B = lengths.size(0)
    T = torch.max(lengths).item()

    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, lengths[i]:] = 1

    return mask.bool()

class TTSDataset(Dataset):
    def __init__(self, 
                 path_to_metadata,
                 sample_rate=22050,
                 n_fft=1024, 
                 window_size=1024, 
                 hop_size=256, 
                 fmin=0,
                 fmax=8000, 
                 num_mels=80, 
                 center=False, 
                 normalized=False, 
                 min_db=-100, 
                 max_scaled_abs=4):
        
        self.metadata = pd.read_csv(path_to_metadata)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax 
        self.num_mels = num_mels
        self.center = center
        self.normalized = normalized
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.transcript_lengths = [len(Tokenizer().encode(t)) for t in self.metadata["normalized_transcript"]]

        self.audio_proc = AudioMelConversions(num_mels=self.num_mels, 
                                              sampling_rate=self.sample_rate, 
                                              n_fft=self.n_fft, 
                                              window_size=self.win_size, 
                                              hop_size=self.hop_size, 
                                              fmin=self.fmin, 
                                              fmax=self.fmax, 
                                              center=self.center,
                                              min_db=self.min_db, 
                                              max_scaled_abs=self.max_scaled_abs)
        
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):

        sample = self.metadata.iloc[idx]
        
        path_to_audio = sample["file_path"]
        transcript = sample["normalized_transcript"]

        audio = load_wav(path_to_audio, sr=self.sample_rate)

        mel = self.audio_proc.audio2mel(audio, do_norm=True)

        return transcript, mel.squeeze(0)

def TTSCollator():

    tokenizer = Tokenizer()

    def _collate_fn(batch):
        
        texts = [tokenizer.encode(b[0]) for b in batch]
        mels = [b[1] for b in batch]
        
        ### Get Lengths of Texts and Mels ###
        input_lengths = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
        output_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)

        ### Sort by Text Length (as we will be using packed tensors later) ###
        input_lengths, sorted_idx = input_lengths.sort(descending=True)
        texts = [texts[i] for i in sorted_idx]
        mels = [mels[i] for i in sorted_idx]
        output_lengths = output_lengths[sorted_idx]

        ### Pad Text ###
        text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)

        ### Pad Mel Sequences ###
        max_target_len = max(output_lengths).item()
        num_mels = mels[0].shape[0]
        
        ### Get gate which tells when to stop decoding. 0 is keep decoding, 1 is stop ###
        mel_padded = torch.zeros((len(mels), num_mels, max_target_len))
        gate_padded = torch.zeros((len(mels), max_target_len))

        for i, mel in enumerate(mels):
            t = mel.shape[1]
            mel_padded[i, :, :t] = mel
            gate_padded[i, t-1:] = 1
        
        mel_padded = mel_padded.transpose(1,2)

        return text_padded, input_lengths, mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths)


    return _collate_fn

class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.sampler = torch.utils.data.SequentialSampler(dataset)
        self.batch_size = batch_size
        self.random_batches = self._make_batches()

    def _make_batches(self):

        indices = [i for i in self.sampler]
        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        random_indices = torch.randperm(len(batches))
        return [batches[i] for i in random_indices]
    
    def __iter__(self):
        for batch in self.random_batches:
            yield batch

    def __len__(self):
        return len(self.random_batches)

if __name__ == "__main__":
    
    

    from torch.utils.data import DataLoader

    dataset = TTSDataset("data/train_metadata.csv")
    
    loader = DataLoader(dataset, collate_fn=TTSCollator(), batch_sampler=BatchSampler(dataset, batch_size=4))
    for sample in loader:
        print(sample[2].min(), sample[2].max())
        break


# class RandomBucketBatchSampler(Sampler):
#     def __init__(self, lengths, batch_size, bucket_size=100, drop_last=False):

#         self.lengths = lengths
#         self.batch_size = batch_size
#         self.bucket_size = bucket_size
#         self.drop_last = drop_last
#         self.buckets = self._create_buckets()

#     def _create_buckets(self):
#         # Sort indices by length
#         sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        
#         # Group into buckets
#         return [
#             sorted_indices[i:i+self.bucket_size]
#             for i in range(0, len(sorted_indices), self.bucket_size)
#         ]

#     def __iter__(self):
#         all_batches = []

#         for bucket in self.buckets:
#             if len(bucket) < self.batch_size and self.drop_last:
#                 continue
            
#             # Shuffle within each bucket
#             random.shuffle(bucket)

#             # Split bucket into batches
#             batches = [
#                 bucket[i:i+self.batch_size]
#                 for i in range(0, len(bucket), self.batch_size)
#             ]
            
#             if self.drop_last:
#                 batches = [b for b in batches if len(b) == self.batch_size]

#             all_batches.extend(batches)

#         # Shuffle batch order across buckets
#         random.shuffle(all_batches)

#         for batch in all_batches:
#             yield batch

#     def __len__(self):
#         count = 0
#         for bucket in self.buckets:
#             n_batches = len(bucket) // self.batch_size
#             if not self.drop_last and len(bucket) % self.batch_size:
#                 n_batches += 1
#             count += n_batches
#         return count
    
# def TTSCollator():

#     tokenizer = Tokenizer()

#     def _collate_fn(batch):
        
#         texts = [tokenizer.encode(b[0]) for b in batch]
#         mels = [b[1] for b in batch]
        
#         ### Get Lengths of Texts and Mels ###
#         input_lengths = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
#         output_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)

#         ### Sort by Text Length (as we will be using packed tensors later) ###
#         input_lengths, sorted_idx = input_lengths.sort(descending=True)
#         texts = [texts[i] for i in sorted_idx]
#         mels = [mels[i] for i in sorted_idx]
#         output_lengths = output_lengths[sorted_idx]

#         ### Pad Text ###
#         text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)

#         ### Pad Mel Sequences ###
#         max_target_len = max(output_lengths).item()
#         num_mels = mels[0].shape[0]
        
#         ### Get gate which tells when to stop decoding. 0 is keep decoding, 1 is stop ###
#         mel_padded = torch.zeros((len(mels), num_mels, max_target_len))
#         gate_padded = torch.zeros((len(mels), max_target_len))

#         for i, mel in enumerate(mels):
#             t = mel.shape[1]
#             mel_padded[i, :, :t] = mel
#             gate_padded[i, t-1:] = 1
        
#         mel_padded = mel_padded.transpose(1,2)

#         # return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
#         return text_padded, input_lengths, mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths)


#     return _collate_fn
        

# if __name__ == "__main__":

#     from torch.utils.data import DataLoader

#     dataset = TTSDataset("data/train_metadata.csv")

#     batch_sampler = RandomBucketBatchSampler(
#         lengths=dataset.transcript_lengths,
#         batch_size=8,
#         bucket_size=100,
#         drop_last=False
#     )

#     dataloader = DataLoader(
#         dataset,
#         batch_sampler=batch_sampler,
#         collate_fn=TTSCollator(),
#         num_workers=32
#     )

#     loader = DataLoader(dataset, batch_size=8, num_workers=32, collate_fn=TTSCollator())

#     for idx, _ in enumerate(dataloader):
#         print(idx)

#     print(len(dataloader))
    
#     # for text_padded, input_lengths, mel_padded, gate_padded, encoder_mask, decoder_mask in dataloader:
#     #     print(gate_padded.sum(axis=-1))
#     #     gate_padded = gate_padded.reshape(-1,1)
#     #     decoder_mask = ~decoder_mask.reshape(-1).bool()

#     #     print(gate_padded[decoder_mask].sum())
#     #     print("---")
       