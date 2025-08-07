import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from librosa.filters import mel

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

class ComputeMelSpectrogram:
    def __init__(self, n_fft, num_mels, sampling_rate, hop_size, window_size, fmin, fmax):
        
        self.n_fft = n_fft 
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.window_size = window_size
        self.fmin = fmin
        self.fmax = fmax

        ### Compute Mel Projection Matrix ###
        self.mel = torch.from_numpy(
            mel(
                sr=sampling_rate, 
                n_fft=n_fft, 
                n_mels=num_mels, 
                fmin=fmin, 
                fmax=fmax
            )
        )
        
        self.hann_window = torch.hann_window(window_size)

    def __call__(self, audio):

        pad = int((self.n_fft-self.hop_size)/2)
        audio = torch.nn.functional.pad(audio, (pad, pad), mode='reflect')

        spec = torch.stft(audio, self.n_fft, hop_length=self.hop_size, win_length=self.window_size, window=self.hann_window.to(audio.device),
                          center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + 1e-9)

        spec = torch.matmul(self.mel.to(audio.device), spec)
        spec = log_scale_spectrograms(spec)

        return spec


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
                 max_audio_magnitude=0.95,):
        
        with open(path_to_manifest, "r") as f:
            self.audio_files = [line.strip() for line in f.readlines()]

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

        mel = compute_mel_spectrogram(audio=audio, 
                                      sampling_rate=self.sampling_rate, 
                                      n_fft=self.n_fft, 
                                      window_size=self.win_size, 
                                      hop_size=self.hop_size, 
                                      fmin=self.fmin, 
                                      fmax=self.fmax, 
                                      num_mels=self.num_mels, 
                                      center=False, 
                                      normalized=False)   

        mel_loss = compute_mel_spectrogram(audio=audio, 
                                           sampling_rate=self.sampling_rate, 
                                           n_fft=self.n_fft, 
                                           window_size=self.win_size, 
                                           hop_size=self.hop_size, 
                                           fmin=self.fmin, 
                                           fmax=self.fmax_loss, 
                                           num_mels=self.num_mels, 
                                           center=False, 
                                           normalized=False)
        
        return (mel.squeeze(0), audio.squeeze(0), mel_loss.squeeze(0))
