__author__ = 'Erdene-Ochir Tuguldur'

import random
import numpy as np

import librosa
import python_speech_features as psf


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
        # audio_duration = len(samples) * 1.0 / sample_rate

        data['samples'] = samples
        data['sample_rate'] = sample_rate

        return data


class LoadMelSpectrogram(object):
    """Loads a mel spectrogram. It assumes that is saved in the same folder like the wav files."""

    def __call__(self, data):
        features = np.load(data['fname'].replace('.wav', '.npy'))

        data = {
            'target': data['text'],
            'target_length': len(data['text']),
            'input': features.astype(np.float32),
            'input_length': features.shape[0]
        }

        return data


class MaskMelSpectrogram(object):
    """Masking the spectrogram aka SpecAugment."""

    def __init__(self, frequency_mask_max_percentage=0.3, time_mask_max_percentage=0.1, probability=1.0):
        self.frequency_mask_probability = frequency_mask_max_percentage
        self.time_mask_probability = time_mask_max_percentage
        self.probability = probability

    def __call__(self, data):
        if random.random() < self.probability:
            mel_spectrogram = data['input']
            tau, nu = mel_spectrogram.shape

            f = random.randint(0, int(self.frequency_mask_probability*nu))
            f0 = random.randint(0, nu - f)
            mel_spectrogram[:, f0:f0 + f] = 0

            t = random.randint(0, int(self.time_mask_probability*tau))
            t0 = random.randint(0, tau - t)
            mel_spectrogram[t0:t0 + t, :] = 0

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


class ComputeMelSpectrogram(object):
    """Computes the mel spectrogram of an audio."""

    def __init__(self, num_features=64):
        self.num_features = num_features
        self.window_size = 20e-3
        self.window_stride = 10e-3

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']

        # T, F
        features = psf.logfbank(signal=samples,
                                samplerate=sample_rate,
                                winlen=self.window_size,
                                winstep=self.window_stride,
                                nfilt=self.num_features,
                                nfft=512,
                                lowfreq=0, highfreq=sample_rate / 2,
                                preemph=0.97)
        # normalize
        m = np.mean(features)
        s = np.std(features)
        features = (features - m) / s

        data = {
            'target': data['text'],
            'target_length': len(data['text']),
            'input': features.astype(np.float32),
            'input_length': features.shape[0]
        }

        return data
