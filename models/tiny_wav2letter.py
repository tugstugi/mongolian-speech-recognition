"""A small wav2letter like network but without GLU."""
__author__ = 'Erdene-Ochir Tuguldur'

import torch.nn as nn
from .layers import *


class TinyWav2Letter(nn.Module):

    def __init__(self, vocab):
        super(TinyWav2Letter, self).__init__()

        self.layers = nn.Sequential(
            C(64, 256, 11, dropout_rate=0.0),  # TODO: stride=2 vs dilation

            C(256, 256, 11, dropout_rate=0.0),
            C(256, 256, 11, dropout_rate=0.0),
            C(256, 256, 11, dropout_rate=0.0),

            C(256, 384, 13, dropout_rate=0.0),

            C(384, 1024, 1, dropout_rate=0.0),

            C(1024, len(vocab), 1)
        )

    def forward(self, x):
        y = self.layers(x)
        return y
