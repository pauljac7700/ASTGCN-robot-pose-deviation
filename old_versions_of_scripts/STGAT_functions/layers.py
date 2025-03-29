# layers.py

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class GatedLinearUnits(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels=16, kernel_size=2, dilation=1, cuda=True, groups=4, activate=False):
        super(GatedLinearUnits, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cuda = cuda
        self.activate = activate

        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation), bias=True, groups=groups))
        nn.init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.conv.bias, 0.1)
        self.gate = weight_norm(nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation), bias=True, groups=groups))
        nn.init.xavier_uniform_(self.gate.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.gate.bias, 0.1)
        self.downsample = weight_norm(nn.Conv2d(in_channels, out_channels, (1, 1), bias=True))
        nn.init.xavier_uniform_(self.downsample.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.downsample.bias, 0.1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.2)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.fill_(0.1)

        self.sigmod = nn.Sigmoid()
        
    def forward(self, X):
        res = X
        gate = X

        # Pad only the width dimension (last dimension)
        X = F.pad(X, ((self.kernel_size-1)*self.dilation, 0, 0, 0))
        out = self.conv(X)
        if self.activate:
            out = torch.tanh(out)

        gate = F.pad(gate, ((self.kernel_size-1)*self.dilation, 0, 0, 0))
        gate = self.gate(gate)
        gate = self.sigmod(gate)

        out = torch.mul(out, gate)
        ones = torch.ones_like(gate)

        if res.shape[1] != out.shape[1]:
            res = self.downsample(res)

        res = torch.mul(res, ones - gate)
        out = out + res
        out = self.relu(self.bn(out))
        return out

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, nhid_channels=128, dropout=0.6, layer=3, cuda=True):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        layers = []
        for i in range(layer):
            if i == 0:
                layers.append(GatedLinearUnits(in_channels, nhid_channels, kernel_size=1, dilation=1, cuda=cuda, groups=1))
            elif i == layer-1:
                layers.append(GatedLinearUnits(nhid_channels, out_channels, kernel_size=1, dilation=1, cuda=cuda, groups=1))
            else:
                layers.append(GatedLinearUnits(nhid_channels, nhid_channels, kernel_size=kernel_size, dilation=2**i, cuda=cuda, groups=1))
        self.units = nn.Sequential(*layers)

    def forward(self, X):
        out = self.units(X) 
        return out