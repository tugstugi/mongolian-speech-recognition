__author__ = 'Erdene-Ochir Tuguldur'

import copy
import numpy as np

import librosa
import python_speech_features as psf


class LoadAudio(object):
    """Loads an audio into a numpy array."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

        self.num_features = 64
        self.window_size = 20e-3
        self.window_stride = 10e-3

    def __call__(self, data):
        fname = data['fname']
        samples, sample_rate = librosa.load(fname, self.sample_rate)
        audio_duration = len(samples) * 1.0 / sample_rate

        # T, F
        features = psf.logfbank(signal=samples,
                                samplerate=sample_rate,
                                winlen=self.window_size,
                                winstep=self.window_stride,
                                nfilt=self.num_features,
                                nfft=512,
                                lowfreq=0, highfreq=sample_rate / 2,
                                preemph=0.97)

        m = np.mean(features)
        s = np.std(features)
        features = (features - m) / s

        data = {
            'target': data['text'],
            'target_length': len(data['text']),
            'input': features,
            'input_length': features.shape[0]
        }
        # data['sample_rate'] = sample_rate
        # data['duration'] = audio_duration

        return data
