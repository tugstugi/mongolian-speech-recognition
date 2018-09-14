#!/usr/bin/env python

"""Read a WAV file and try to recognize the speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import argparse
import torch

from datasets import LoadAudio
from datasets.mb_speech import vocab, idx2char
from models import TinyWav2Letter
from utils import load_checkpoint

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--checkpoint", type=str, required=True, help='checkpoint file to test')
parser.add_argument("audio", help='a WAV file')
args = parser.parse_args()

data = {
    'fname': args.audio,
    'text': ''
}
data = LoadAudio()(data)

model = TinyWav2Letter(vocab)
load_checkpoint(args.checkpoint, model, optimizer=None)
model.eval()
model.cpu()

input = torch.from_numpy(data['input']).unsqueeze(0)
inputs = input.permute(0, 2, 1)

torch.set_grad_enabled(False)
outputs = model(inputs)
outputs = outputs.permute(2, 0, 1)


def to_text(tensor, max_length=None, remove_repetitions=False):
    sentence = ''
    sequence = tensor.cpu().detach().numpy()
    for i in range(len(sequence)):
        if max_length is not None and i >= max_length:
            continue
        char = idx2char[sequence[i]]
        if char != 'B':  # ignore blank
            if remove_repetitions and i != 0 and char == idx2char[sequence[i - 1]]:
                pass
            else:
                sentence = sentence + char
    return sentence


prediction = outputs.softmax(2).max(2)[1]
predicted_text = to_text(prediction[:, 0], remove_repetitions=True)

print("Predicted:")
print(predicted_text)
