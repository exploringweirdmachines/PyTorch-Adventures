import librosa

import matplotlib.pyplot as plt
import torch
from torchaudio.transforms import MelSpectrogram

n_fft = 1024
win_len = 1024
hop_len = 256
n_mels = 80
sample_rate = 8000

path = '/mnt/datadrive/data/LJSpeech-1.1/wavs/LJ050-0272.wav'

waveform, sample_rate = librosa.load(path, sr=sample_rate)
waveform = torch.Tensor(waveform)

torchaudio_melspec = MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_len,
    hop_length=hop_len,
    center=True,
    pad_mode="constant",
    power=2.0,
    norm='slaney',
    mel_scale='slaney',
    n_mels=n_mels,
)(waveform)

librosa_melspec = librosa.feature.melspectrogram(
    y=waveform.numpy(),
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_len,
    win_length=win_len,
    center=True,
    pad_mode="constant",
    power=2.0,
    n_mels=n_mels,
    norm='slaney',
    # htk=True, # default is False
)

mse = ((torchaudio_melspec - librosa_melspec) ** 2).mean()

print(f'MSE:\t{mse}')

fig, axs = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle('Mel Spectrogram')

axs[0].set_title('torchaudio')
axs[0].set_ylabel('mel bin')
axs[0].set_xlabel('frame')
axs[0].imshow(librosa.power_to_db(torchaudio_melspec), aspect='auto')

axs[1].set_title('librosa')
axs[1].set_ylabel('mel bin')
axs[1].set_xlabel('frame')
axs[1].imshow(librosa.power_to_db(librosa_melspec), aspect='auto')

plt.show()