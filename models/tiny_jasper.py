"""Modified version of "Jasper: An End-to-End Convolutional Neural Acoustic Model".
Residual connections are implemented differently than the original.

See:
https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/jasper10x5_LibriSpeech_nvgrad.py
"""
__author__ = 'Erdene-Ochir Tuguldur'

import torch.nn as nn
from .layers import *


class TinyJasper(nn.Module):

    def __init__(self, vocab):
        super(TinyJasper, self).__init__()

        self.first_layer = C(64, 256, 11, stride=2, dropout_rate=0.2)

        self.B1 = nn.Sequential(
            C(256, 256, 11, dropout_rate=0.2),
            C(256, 256, 11, dropout_rate=0.2),
            C(256, 256, 11, dropout_rate=0.2),
        )

        self.B2 = nn.Sequential(
            C(256, 384, 13, dropout_rate=0.2),
            C(384, 384, 13, dropout_rate=0.2),
            C(384, 384, 13, dropout_rate=0.2),
        )
        self.r2 = nn.Conv1d(256, 384, 1)

        self.B3 = nn.Sequential(
            C(384, 512, 17, dropout_rate=0.2),
            C(512, 512, 17, dropout_rate=0.2),
            C(512, 512, 17, dropout_rate=0.2),
        )
        self.r3 = nn.Conv1d(384, 512, 1)

        self.B4 = nn.Sequential(
            C(512, 640, 21, dropout_rate=0.3),
            # C(640, 640, 21, dropout_rate=0.3),
            # C(640, 640, 21, dropout_rate=0.3),
        )
        self.B5 = nn.Sequential(
            C(640, 768, 25, dropout_rate=0.3),
            # C(768, 768, 25, dropout_rate=0.3),
            # C(768, 768, 25, dropout_rate=0.3),
        )
        self.r4_5 = nn.Conv1d(512, 768, 1)

        self.last_layer = nn.Sequential(
            C(768, 896, 29, dropout_rate=0.4, dilation=2),
            C(896, 1024, 1, dropout_rate=0.4),
            C(1024, len(vocab), 1)
        )

    def forward(self, x):
        y = self.first_layer(x)

        y = self.B1(y) + y
        y = self.B2(y) + self.r2(y)
        y = self.B3(y) + self.r3(y)
        y = self.B5(self.B4(y)) + self.r4_5(y)

        y = self.last_layer(y)
        # BxCxT
        return y
