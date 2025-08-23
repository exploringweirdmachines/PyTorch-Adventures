import random
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from tokenizer import Tokenizer

def log_scale_spectrograms(spectrogram, clip_val=1e-5):
    return torch.log(torch.clamp(spectrogram, min=clip_val))

def compute_mel_spectrogram(audio, 
                            sampling_rate, 
                            n_fft, 
                            window_size, 
                            hop_size, 
                            fmin, 
                            fmax, 
                            num_mels, 
                            center=False, 
                            normalized=False):

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        win_length=window_size,
        hop_length=hop_size,
        f_min=fmin,
        f_max=fmax,
        n_mels=num_mels,
        window_fn=torch.hann_window,
        center=center,
        power=1.0,
        pad_mode='reflect',
        norm='slaney',
        mel_scale='slaney',
        normalized=normalized
    ).to(audio.device)
        
    pad = ((n_fft - hop_size) // 2)
    reflect_for_mel_audio = F.pad(audio, (pad, pad), mode='reflect')

    mel = mel_transform(reflect_for_mel_audio)

    mel = log_scale_spectrograms(mel)

    return mel

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
                 normalized=False):
        
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

        self.transcript_lengths = [len(Tokenizer().encode(t)) for t in self.metadata["normalized_transcript"]]
        
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):

        sample = self.metadata.iloc[idx]
        
        path_to_audio = sample["file_path"]
        transcript = sample["normalized_transcript"]

        audio, sr = torchaudio.load(path_to_audio)

        if sr != self.sample_rate:
            audio = TF.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

        mel = compute_mel_spectrogram(audio=audio, 
                                      sampling_rate=self.sample_rate, 
                                      n_fft=self.n_fft, 
                                      window_size=self.win_size, 
                                      hop_size=self.hop_size, 
                                      fmin=self.fmin, 
                                      fmax=self.fmax, 
                                      num_mels=self.num_mels, 
                                      center=False, 
                                      normalized=False)
        
        return transcript, mel.squeeze(0)

def build_padding_mask(lengths):

    B = len(lengths)
    T = torch.max(lengths).item()

    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, lengths[i]:] = 1
    
    return mask

class RandomBucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, bucket_size=100, drop_last=False):

        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.drop_last = drop_last
        self.buckets = self._create_buckets()

    def _create_buckets(self):
        # Sort indices by length
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        
        # Group into buckets
        return [
            sorted_indices[i:i+self.bucket_size]
            for i in range(0, len(sorted_indices), self.bucket_size)
        ]

    def __iter__(self):
        all_batches = []

        for bucket in self.buckets:
            if len(bucket) < self.batch_size and self.drop_last:
                continue
            
            # Shuffle within each bucket
            random.shuffle(bucket)

            # Split bucket into batches
            batches = [
                bucket[i:i+self.batch_size]
                for i in range(0, len(bucket), self.batch_size)
            ]
            
            if self.drop_last:
                batches = [b for b in batches if len(b) == self.batch_size]

            all_batches.extend(batches)

        # Shuffle batch order across buckets
        random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        count = 0
        for bucket in self.buckets:
            n_batches = len(bucket) // self.batch_size
            if not self.drop_last and len(bucket) % self.batch_size:
                n_batches += 1
            count += n_batches
        return count
    
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

        # return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
        return text_padded, input_lengths, mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths)


    return _collate_fn
        

if __name__ == "__main__":

    from torch.utils.data import DataLoader

    dataset = TTSDataset("data/train_metadata.csv")

    batch_sampler = RandomBucketBatchSampler(
        lengths=dataset.transcript_lengths,
        batch_size=8,
        bucket_size=100,
        drop_last=False
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=TTSCollator(),
        num_workers=32
    )

    loader = DataLoader(dataset, batch_size=8, num_workers=32, collate_fn=TTSCollator())

    for idx, _ in enumerate(dataloader):
        print(idx)

    print(len(dataloader))
    
    # for text_padded, input_lengths, mel_padded, gate_padded, encoder_mask, decoder_mask in dataloader:
    #     print(gate_padded.sum(axis=-1))
    #     gate_padded = gate_padded.reshape(-1,1)
    #     decoder_mask = ~decoder_mask.reshape(-1).bool()

    #     print(gate_padded[decoder_mask].sum())
    #     print("---")
       