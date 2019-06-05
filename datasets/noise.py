"""Noise dataset copied from: https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/generator.py"""

import random
import numpy as np
from acoustics.generator import white, pink, violet, blue, brown

from torch.utils.data import Dataset
from .transforms import Compose, ComputeMagSpectrogram


class NoiseDataset(Dataset):

    def __init__(self, size=1000, sample_rate=16000, transform=None):
        self.size = size
        self.sample_rate = sample_rate
        self.transform = None if transform is None else Compose(transform.transforms[1:])

        n = 1000000

        compute_mag_spectrogram = ComputeMagSpectrogram()
        self.n_fft = compute_mag_spectrogram.n_fft

        def compute_spectrogram(sample):
            clipped_sample = np.clip(sample/5, -1, 1).astype(np.float32)  # amplitude is around (-0.7, 0.7)
            return compute_mag_spectrogram({'text': '', 'samples': clipped_sample, 'sample_rate': sample_rate})['input']

        self.noises = [
            compute_spectrogram(white(n)),
            compute_spectrogram(pink(n)),
            compute_spectrogram(violet(n))
        ]

    def get_random_noise(self, width):
        noise = random.choice(self.noises)
        _, t = noise.shape
        start_index = random.randint(0, t - width)
        return noise[:, start_index:start_index + width]

    def __getitem__(self, index):
        features = self.get_random_noise(random.randint(200, 500))
        data = {
            'target': np.array([1], dtype=np.int),
            'target_length': 1,
            'input': features.astype(np.float32),
            'input_length': features.shape[1],
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return self.size
