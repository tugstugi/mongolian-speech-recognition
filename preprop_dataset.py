#!/usr/bin/env python
"""Read WAV files and compute spectrograms and save them in the same folder."""
__author__ = 'Erdene-Ochir Tuguldur'

import argparse
from tqdm import *
from joblib import Parallel, delayed
import numpy as np

from torch.utils.data import ConcatDataset
from datasets import Compose, LoadAudio, ComputeMagSpectrogram

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset",
                    choices=['librispeech', 'mbspeech', 'bolorspeech', 'kazakh335h', 'germanspeech', 'backgroundsounds'],
                    default='bolorspeech', help='dataset name')
parser.add_argument("--jobs", type=int, default=1, help="parallel jobs")
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
elif args.dataset == 'backgroundsounds':
    from datasets.background_sounds import BackgroundSounds
    dataset = BackgroundSounds(is_random=False)
elif args.dataset == 'bolorspeech':
    from datasets.bolor_speech import BolorSpeech
    dataset = ConcatDataset([
        BolorSpeech(name='train'),
        BolorSpeech(name='train2'),
        BolorSpeech(name='test'),
        BolorSpeech(name='demo'),
        BolorSpeech(name='annotation'),
        BolorSpeech(name='annotation-1111')
    ])
elif args.dataset == 'kazakh335h':
    from datasets.kazakh335h_speech import Kazakh335hSpeech
    dataset = ConcatDataset([
        Kazakh335hSpeech(name='test'),
        Kazakh335hSpeech(name='dev'),
        Kazakh335hSpeech(name='train')
    ])
elif args.dataset == 'germanspeech':
    from datasets.german_speech import GermanSpeech
    dataset = ConcatDataset([
        GermanSpeech(name='train'),
        GermanSpeech(name='dev_swc'),
        GermanSpeech(name='dev_tuda'),
        GermanSpeech(name='dev_voxforge'),
        GermanSpeech(name='test_swc', max_duration=40),
        GermanSpeech(name='test_tuda', max_duration=40),
        GermanSpeech(name='test_voxforge', max_duration=40)
    ])
else:
    print("unknown dataset!")
    import sys
    sys.exit(1)


transform=Compose([LoadAudio(), ComputeMagSpectrogram()])


def preprocess(data):
    fname = data['fname']
    data = transform(data)
    mel_spectrogram = data['input']
    np.save(fname.replace('.wav', '.npy'), mel_spectrogram)


Parallel(n_jobs=args.jobs)(delayed(preprocess)(d) for d in tqdm(dataset))
