#!/usr/bin/env python

"""Record and try to recognize the speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import argparse

import sounddevice as sd

from datasets import Compose, ComputeMagSpectrogram, ComputeMelSpectrogramFromMagSpectrogram
from transcribe import transcribe

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help='checkpoint file to test')
    parser.add_argument("--model", choices=['crnn', 'quartznet5x5', 'quartznet10x5', 'quartznet15x5'], default='crnn',
                        help='choices of neural network')
    parser.add_argument("--normalize", choices=['all_features', 'per_feature'], default='all_features',
                        help="feature normalization")
    parser.add_argument("--lm", type=str, required=False, help='link to KenLM 5-gram binary language model')
    parser.add_argument("--alpha", type=float, default=0.3, help='alpha for CTC decode')
    parser.add_argument("--beta", type=float, default=1.85, help='beta for CTC decode')
    args = parser.parse_args()
    print(args)

    num_features = 64
    eps = 2 ** -24
    if args.model == 'crnn':
        # CRNN supports only 32 features
        num_features = 32
        eps = 1e-20

    duration = 5.0  # s
    sample_rate = 16000
    print("recording %0.1fs audio..." % duration)
    recorded_audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, blocking=True)
    print("recorded, replaying it before doing speech recognition...")
    sd.play(recorded_audio, samplerate=sample_rate, blocking=True)

    transform = Compose([ComputeMagSpectrogram(),
                         ComputeMelSpectrogramFromMagSpectrogram(num_features=num_features,
                                                                 normalize=args.normalize, eps=eps)])

    transcribe(transform({
        'samples': recorded_audio,
        'sample_rate': sample_rate,
        'text': ''
    }), num_features, args)
