import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas as pd
import torchaudio

class AudioMnistDataset(Dataset):
    def __init__(self,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 annotation_path,
                 device):
        
        self.annotations = pd.read_csv(annotation_path)
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        audio_sample_path = self.annotations.iloc[index,1]
        audio_sample_label = self.annotations.iloc[index,0]
        
        signal,sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)

        signal = self._resample_if_necessary(signal,sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        signal = self.transformation(signal)

        return signal,audio_sample_label
    
    def _resample_if_necessary(self,signal,sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self,signal):
        if signal.shape[0]>1:
            signal = torch.mean(signal,dim=0,keepdim=True)
        return signal

    def _cut_if_necessary(self,signal):
        if signal.shape[1] > self.num_samples:
            signal[:,:self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self,signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0,num_missing_samples)
            signal = torch.nn.functional.pad(signal,last_dim_padding)
        return signal

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('DEVICE:',device)

    ANNOT_PATH = r'./annotation.csv'
    SAMPLE_RATE = 22050   
    NUM_SAMPLES =22050


    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #     sample_rate = SAMPLE_RATE,
    #     n_fft = 512,    # n_fft 값을 조정하여 필요한 빈도 해상도와 계산 비용을 조절할 수 있다
    #     win_length=400,
    #     hop_length=160,  # 이전 프레임과 다음 프레임 사이에서 겹치는 부분의 길이를 결정하는 파라미터
    #     n_mels=80        # 
    #     )

    transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate = SAMPLE_RATE,
            n_fft= 512,
            win_length= 512,
            hop_length= 160,
            n_mels= 80),
       torchaudio.transforms.AmplitudeToDB()
    )

    temp = AudioMnistDataset(transformation=transforms,
                             target_sample_rate=22050,
                             num_samples=22050,
                             annotation_path=ANNOT_PATH,
                             device=device)
