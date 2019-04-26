#!/usr/bin/env python
"""Read WAV files and save them mel spectrogram in the same folder."""
__author__ = 'Erdene-Ochir Tuguldur'

import numpy as np
from tqdm import *
from datasets import Compose, LoadAudio, ComputeMelSpectrogram
from datasets.mb_speech import MBSpeech as SpeechDataset

transform=Compose([LoadAudio(), ComputeMelSpectrogram()])
mbspeech = SpeechDataset()

for data in tqdm(mbspeech):
    fname = data['fname']
    data = transform(data)
    mel_spectrogram = data['input']
    np.save(fname.replace('.wav', '.npy'), mel_spectrogram)
