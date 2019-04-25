#!/usr/bin/env python

"""Record and try to recognize the speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import argparse

import sounddevice as sd

from datasets import Compose, ExtractSpeechFeatures
from eval import transcribe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help='checkpoint file to test')
    parser.add_argument("--lm", type=str, required=False, help='link to KenLM 5-gram binary language model')
    parser.add_argument("--alpha", type=float, default=0.3, help='alpha for CTC decode')
    parser.add_argument("--beta", type=float, default=1.85, help='beta for CTC decode')
    args = parser.parse_args()

    duration = 5.0  #s
    sample_rate = 16000
    print("recording %0.1fs audio..." % duration)
    recorded_audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, blocking=True)
    print("recorded, replaying it before doing speech recognition...")
    sd.play(recorded_audio, samplerate=sample_rate, blocking=True)

    data = {
        'samples': recorded_audio,
        'sample_rate': sample_rate,
        'text': ''
    }
    data = Compose([ExtractSpeechFeatures()])(data)

    result = transcribe(data, args)

    print("Predicted:")
    print(result)
