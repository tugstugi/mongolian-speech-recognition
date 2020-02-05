"""Data loader for the background sounds which is used as a noise."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import numpy as np
import random
from torch.utils.data import Dataset

vocab = "B "  # B: blank
char2idx = {char: idx for idx, char in enumerate(vocab)}


class BackgroundSounds(Dataset):

    def __init__(self, size=5000, max_duration=10, transform=None, is_random=True):
        self.size = size
        self.is_random = is_random
        self.transform = transform

        # CSV file contains fname, duration_ms
        datasets_path = os.path.dirname(os.path.realpath(__file__))
        csv_file = os.path.join(datasets_path, 'background_sounds.csv')
        lines = open(csv_file, 'r').readlines()

        self.all_background_sounds = []
        self.fnames = []
        for l in lines:
            fname, duration = l.strip().split(',')
            self.all_background_sounds.append(fname)
            duration = float(duration)
            if duration <= max_duration:
                self.fnames.append(os.path.join(datasets_path, fname))
            # self.all_background_sounds.append([fname, 1 + int((len(y) - frame_length) / hop_length)])

    def __getitem__(self, index):
        text = [char2idx[' ']]  # only a single whitespace because it is noise

        data = {
            'fname': random.choice(self.fnames) if self.is_random else self.fnames[index],
            'text': np.array(text, dtype=np.int)
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return self.size if self.is_random else len(self.fnames)


if __name__ == '__main__':
    # from transforms import *
    # transform = Compose([LoadMagSpectrogram()])
    d = BackgroundSounds()
    print(len(d))
    print(d[10])
