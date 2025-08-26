import random
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import librosa

import numpy as np

def load_wav(path_to_audio, sr=22050):
    audio, orig_sr = torchaudio.load(path_to_audio)

    if sr != orig_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sr)

    return audio.squeeze(0)

def amp_to_db(x, min_db=-100):
    ### Forces min DB to be -100
    ### 20 * torch.log10(1e-5) = 20 * -5 = -100
    clip_val = 10 ** (min_db / 20)
    return 20 * torch.log10(torch.clamp(x, min=clip_val))

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

def pad_for_mel(x, n_fft, hop_size):
    pad = ((n_fft - hop_size) // 2)
    return F.pad(x, (pad, pad), mode="reflect")

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

        mel = amp_to_db(mel, self.min_db)
        
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

class MelDataset(Dataset):
    def __init__(self, 
                 path_to_manifest, 
                 segment_size=8192, 
                 n_fft=1024, 
                 num_mels=80,
                 hop_size=256, 
                 window_size=1024, 
                 sampling_rate=22050,  
                 fmin=0, 
                 fmax=8000, 
                 fmax_loss=None,
                 max_audio_magnitude=0.95, 
                 min_db=-100, 
                 max_scaled_abs=4):

        self.audio_files = list(pd.read_csv(path_to_manifest)["file_path"])

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = window_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.max_audio_magnitude = max_audio_magnitude

        common_args = dict(
            num_mels=num_mels, 
            sampling_rate=sampling_rate,
            n_fft=n_fft, 
            window_size=window_size, 
            hop_size=hop_size, 
            fmin=fmin, 
            center=False,
            min_db=min_db, 
            max_scaled_abs=max_scaled_abs
        )

        self.audio_mel_conv = AudioMelConversions(
            **common_args,
            fmax=fmax,
        )

        self.audio_mel_conv_loss = AudioMelConversions(
            **common_args,
            fmax=None,
        )
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):

        filename = self.audio_files[index]

        audio, sampling_rate = torchaudio.load(filename)
        
        audio = audio / torch.max(audio) * self.max_audio_magnitude

        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        if audio.shape[1] >= self.segment_size:
            max_audio_start = audio.shape[1] - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start+self.segment_size]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        audio_padded_for_mel = pad_for_mel(audio, self.n_fft, self.hop_size)

        mel = self.audio_mel_conv.audio2mel(audio_padded_for_mel, do_norm=True)
        mel_loss = self.audio_mel_conv_loss.audio2mel(audio_padded_for_mel, do_norm=True)

        return (mel.squeeze(0), audio, mel_loss.squeeze(0))


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    dataset = MelDataset(path_to_manifest="data/train_metadata.csv")

    loader = DataLoader(dataset, batch_size=4)

    for a, b, c in loader:
        pass
