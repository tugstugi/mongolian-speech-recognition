"""Common layers."""
__author__ = 'Erdene-Ochir Tuguldur'
__all__ = ['C']

import torch
import torch.nn as nn
import torch.nn.functional as F


class C(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, activation='relu', dropout_rate=0.0):
        """1D Convolution with the batch normalization and RELU."""
        super(C, self).__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate

        assert 1 <= stride <= 2
        if dilation > 1:
            assert stride == 1
            padding = (kernel_size - 1) * dilation // 2
        else:
            padding = (kernel_size - stride + 1) // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation)
        nn.init.xavier_uniform_(self.conv.weight, nn.init.calculate_gain('relu'))

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)

        if self.activation == 'relu':
            y = F.relu(y, inplace=True)
            # OpenSeq2Seq uses max clamping instead of gradient clipping
            # y = torch.clamp(y, min=0.0, max=20.0)  # like RELU but clamp at 20.0

        if self.dropout_rate > 0:
            y = F.dropout(y, p=self.dropout_rate, training=self.training, inplace=False)
        return y


if __name__ == '__main__':
    import numpy as np
    import torch
    c = C(1, 5, 3, 3, 0.0)
    print(c(torch.from_numpy(np.zeros((1, 1, 10))).float()))
