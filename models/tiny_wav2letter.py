"""A small wav2letter like network but without GLU.

Instead of the original Facebook implementation:
https://github.com/facebookresearch/wav2letter/blob/master/arch/librispeech-glu-highdropout
, we are going to use the modification introduced by the openseq2seq project:
https://github.com/NVIDIA/OpenSeq2Seq/blob/18.09/example_configs/speech2text/w2lplus_large_8gpus.py
"""
__author__ = 'Erdene-Ochir Tuguldur'

import torch.nn as nn
from .layers import *


class TinyWav2Letter(nn.Module):

    def __init__(self, vocab):
        super(TinyWav2Letter, self).__init__()

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
        self.B3 = nn.Sequential(
            C(384, 512, 17, dropout_rate=0.2),
            C(512, 512, 17, dropout_rate=0.2),
            C(512, 512, 17, dropout_rate=0.2),
        )
        self.B4 = nn.Sequential(
            C(512, 640, 21, dropout_rate=0.3),
            # C(640, 640, 21, dropout_rate=0.3),
            # C(640, 640, 21, dropout_rate=0.3),
        )
        self.B5 = nn.Sequential(
            # C(640, 768, 25, dropout_rate=0.3),
            # C(768, 768, 25, dropout_rate=0.3),
            # C(768, 768, 25, dropout_rate=0.3),
        )

        self.last_layer = nn.Sequential(
            # TODO: dropout_rate=0.4 and dilation=2
            C(640, 896, 29, dropout_rate=0.3, dilation=1),
            C(896, 1024, 1, dropout_rate=0.4),
            C(1024, len(vocab), 1)
        )

    def forward(self, x):
        y = self.first_layer(x)

        y = self.B1(y)
        y = self.B2(y)
        y = self.B3(y)
        y = self.B4(y)
        y = self.B5(y)

        y = self.last_layer(y)
        # BxCxT
        return y
