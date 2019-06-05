#!/usr/bin/env python
"""Read WAV files and compute spectrograms and save them in the same folder."""
__author__ = 'Erdene-Ochir Tuguldur'

import argparse
from tqdm import *
import numpy as np

from torch.utils.data import ConcatDataset
from datasets import Compose, LoadAudio, ComputeMagSpectrogram

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset",
                    choices=['librispeech', 'mbspeech', 'bolorspeech'], default='bolorspeech', help='dataset name')
args = parser.parse_args()

if args.dataset == 'mbspeech':
    from datasets.mb_speech import MBSpeech
    dataset = MBSpeech()
elif args.dataset == 'librispeech':
    from datasets.libri_speech import LibriSpeech
    dataset = ConcatDataset([
        LibriSpeech(name='train-clean-100'),
        LibriSpeech(name='train-clean-360'),
        LibriSpeech(name='train-other-500'),
        LibriSpeech(name='dev-clean',)
    ])
else:
    from datasets.bolor_speech import BolorSpeech
    dataset = ConcatDataset([
        BolorSpeech(name='train'),
        BolorSpeech(name='test')
    ])


transform=Compose([LoadAudio(), ComputeMagSpectrogram()])
for data in tqdm(dataset):
    fname = data['fname']
    data = transform(data)
    mel_spectrogram = data['input']
    np.save(fname.replace('.wav', '.npy'), mel_spectrogram)
