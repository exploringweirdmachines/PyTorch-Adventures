import random
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset
from librosa.filters import mel
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
        
        ### Generate Mask for Padding ###
        encoder_mask = build_padding_mask(input_lengths)
        decoder_mask = build_padding_mask(output_lengths)

        return text_padded, input_lengths, mel_padded, gate_padded, encoder_mask, decoder_mask


    return _collate_fn
        

if __name__ == "__main__":

    dataset = TTSDataset("data/test_metadata.csv")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=TTSCollator())
    for sample in loader:
        break