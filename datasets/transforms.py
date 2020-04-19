__author__ = 'Erdene-Ochir Tuguldur'

import random
import numpy as np
import cv2

import librosa
import albumentations as album


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class LoadAudio(object):
    """Loads an audio into a numpy array."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, data):
        samples, sample_rate = librosa.load(data['fname'], self.sample_rate)

        data['samples'] = samples
        data['sample_rate'] = sample_rate

        return data


class LoadMagSpectrogram(object):
    """Loads a spectrogram. It assumes that is saved in the same folder like the wav files."""

    def __init__(self, sample_rate=16000, n_fft=512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft

    def __call__(self, data):
        # F,T
        features = np.load(data['fname'].replace('.wav', '.npy'))

        data = {
            'target': data['text'],
            'target_length': len(data['text']),
            'input': features.astype(np.float32),
            'input_length': features.shape[1],
            'n_fft': self.n_fft,
            'sample_rate': self.sample_rate,
        }

        return data


class AddNoiseToMagSpectrogram(object):
    """Add noise to a mag spectrogram."""

    def __init__(self, noise, probability=0.5):
        self.probability = probability
        self.noise = noise

    def __call__(self, data):
        if random.random() < self.probability:
            spectrogram = data['input']
            _, t = spectrogram.shape
            dither = random.uniform(1e-5, 1e-3)
            spectrogram = spectrogram + dither * self.noise.get_random_noise(t)
            data['input'] = spectrogram.astype(np.float32)
        return data


class MaskSpectrogram(object):
    """Masking a spectrogram aka SpecAugment."""

    def __init__(self, frequency_mask_max_percentage=0.3, time_mask_max_percentage=0.1, probability=1.0):
        self.frequency_mask_probability = frequency_mask_max_percentage
        self.time_mask_probability = time_mask_max_percentage
        self.probability = probability

    def __call__(self, data):
        if random.random() < self.probability:
            spectrogram = data['input']
            nu, tau = spectrogram.shape

            f = random.randint(0, int(self.frequency_mask_probability*nu))
            f0 = random.randint(0, nu - f)
            spectrogram[f0:f0 + f, :] = 0

            t = random.randint(0, int(self.time_mask_probability*tau))
            t0 = random.randint(0, tau - t)
            spectrogram[:, t0:t0 + t] = 0

            data['input'] = spectrogram

        return data


class ShiftSpectrogramAlongTimeAxis(object):
    """Shift a spectrogram along the time axis."""

    def __init__(self, time_shift_max_percentage=0.1, probability=0.5):
        self.time_shift_max_percentage = time_shift_max_percentage
        self.probability = probability

    @staticmethod
    def shift(spectrogram, d):
        if d != 0:
            spectrogram = np.roll(spectrogram, d, 1)
            if d > 0:
                spectrogram[:, :d] = 0
            else:
                spectrogram[:, d:] = 0
        return spectrogram

    def __call__(self, data):
        if random.random() < self.probability:
            spectrogram = data['input']
            nu, tau = spectrogram.shape

            d = random.randint(-int(self.time_shift_max_percentage * tau), int(self.time_shift_max_percentage * tau))
            data['input'] = self.shift(spectrogram, d)

        return data


class ShiftSpectrogramAlongFrequencyAxis(object):
    """Shift a spectrogram along the frequency axis."""

    def __init__(self, frequency_shift_max_percentage=0.1, probability=0.5):
        self.frequency_shift_max_percentage = frequency_shift_max_percentage
        self.probability = probability

    @staticmethod
    def shift(spectrogram, d):
        if d != 0:
            spectrogram = np.roll(spectrogram, d, 0)
            if d > 0:
                spectrogram[:d, :] = 0
            else:
                spectrogram[d:, :] = 0
        return spectrogram

    def __call__(self, data):
        if random.random() < self.probability:
            spectrogram = data['input']
            nu, tau = spectrogram.shape

            d = random.randint(-int(self.frequency_shift_max_percentage * nu),
                               int(self.frequency_shift_max_percentage * nu))
            data['input'] = self.shift(spectrogram, d)

        return data


class ApplyAlbumentations(object):
    """Apply transforms from Albumentations."""

    def __init__(self, a_transform):
        self.a_transform = a_transform

    def __call__(self, data):
        data['input'] = self.a_transform(image=data['input'])['image']
        return data


class TimeScaleSpectrogram(object):
    """Scaling a spectrogram in the time axis."""

    def __init__(self, max_scale=0.2, probability=0.5):
        self.max_scale = max_scale
        self.probability = probability

    def __call__(self, data):
        if random.random() < self.probability:
            num_features, t = data['input'].shape
            scale = random.uniform(-self.max_scale, self.max_scale)
            data['input'] = cv2.resize(data['input'],
                                       (int(round((1 + scale) * t)), num_features), interpolation=cv2.INTER_LINEAR)
            data['input_length'] = data['input'].shape[-1]
        return data


class SpeedChange(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""

    def __init__(self, max_scale=0.2, probability=0.5):
        self.max_scale = max_scale
        self.probability = probability

    def __call__(self, data):
        if random.random() < self.probability:
            samples = data['samples']

            scale = random.uniform(-self.max_scale, self.max_scale)
            speed_fac = 1.0 / (1 + scale)
            data['samples'] = np.interp(np.arange(0, len(samples), speed_fac),
                                        np.arange(0, len(samples)), samples).astype(np.float32)

        return data


class ComputeMagSpectrogram(object):
    """Computes the magnitude spectrogram of an audio."""

    def __init__(self, n_fft=512, win_length=20e-3, hop_length=10e-3, center=False):
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center

    @staticmethod
    def preemphasis(samples, coeff=0.97):
        return np.append(samples[0], samples[1:] - coeff * samples[:-1])

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']

        samples = self.preemphasis(samples, coeff=0.97)
        stft = librosa.stft(samples, n_fft=self.n_fft,
                            win_length=int(self.win_length*sample_rate),
                            hop_length=int(self.hop_length*sample_rate),
                            center=self.center)
        # F, T
        features = np.abs(stft)

        data = {
            'target': data['text'],
            'target_length': len(data['text']),
            'input': features.astype(np.float32),
            'input_length': features.shape[1],
            'n_fft': self.n_fft,
            'sample_rate': sample_rate
        }

        return data


class ComputeMelSpectrogramFromMagSpectrogram(object):
    """Computes the mel spectrogram from a magnitude spectrogram."""

    def __init__(self, num_features=32, normalize='all_features', eps=1e-20):
        self.num_features = num_features
        self.mel_basis = None
        assert normalize in ['all_features', 'per_feature']
        self.normalize = normalize
        self.eps = eps

    def __call__(self, data):
        if self.mel_basis is None:
            sample_rate = data['sample_rate']
            self.mel_basis = librosa.filters.mel(sr=sample_rate,
                                                 n_fft=data['n_fft'],
                                                 n_mels=self.num_features,
                                                 fmin=0,
                                                 fmax=sample_rate/2,
                                                 htk=False)
        mag = data['input']
        # features = librosa.power_to_db(np.dot(self.mel_basis, mag*mag), ref=np.max)
        features = np.log(np.dot(self.mel_basis, mag*mag) + self.eps)

        # normalize
        if self.normalize == 'all_features':
            m = np.mean(features)
            s = np.std(features) #+ 1e-5
            features = (features - m) / s
        elif self.normalize == 'per_feature':
            m = np.mean(features, axis=1, keepdims=True)
            s = np.std(features, axis=1, keepdims=True) + 1e-5
            features = (features - m) / s

        data['input'] = features.astype(np.float32)
        return data


class DistortMagSpectrogram(object):
    """Distorts a magnitude spectrogram."""

    def __init__(self, num_steps=10, distort_limit=0.4, probability=0.5, min_length=200):
        self.min_length = min_length
        self.transform = album.GridDistortion(num_steps=num_steps, distort_limit=distort_limit, p=probability,
                                              interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT,
                                              value=0)

    def __call__(self, data):
        if data['input_length'] >= self.min_length:
            data['input'] = self.transform(image=data['input'])['image']
        return data

