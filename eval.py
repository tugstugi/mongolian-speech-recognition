#!/usr/bin/env python

"""Read a WAV file and try to recognize the speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import argparse
import torch

from datasets import Compose, LoadAudio, ExtractSpeechFeatures
from datasets.mb_speech import vocab, idx2char
from models import TinyWav2Letter
from utils import load_checkpoint

from decoder import GreedyDecoder


def recognize(checkpoint_file, data):
    model = TinyWav2Letter(vocab)
    load_checkpoint(checkpoint_file, model, optimizer=None)
    model.eval()
    model.cpu()

    inputs = torch.from_numpy(data['input']).unsqueeze(0)
    inputs = inputs.permute(0, 2, 1)

    torch.set_grad_enabled(False)
    outputs = model(inputs)
    outputs = outputs.permute(2, 0, 1)

    decoder = GreedyDecoder(labels=vocab)
    decoded_output, _ = decoder.decode(outputs.softmax(2).permute(1, 0, 2))

    return decoded_output[0][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help='checkpoint file to test')
    parser.add_argument("audio", help='a WAV file')
    args = parser.parse_args()

    data = {
        'fname': args.audio,
        'text': ''
    }
    data = Compose([LoadAudio(), ExtractSpeechFeatures()])(data)

    result = recognize(args.checkpoint, data)

    print("Predicted:")
    print(result)
