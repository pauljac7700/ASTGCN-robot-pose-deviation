import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from lib.STGAT_functions.layers import TimeBlock
from lib.STGAT_functions.readout import AvgReadout
from lib.STGAT_functions.discriminator import Discriminator

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_nodes = num_nodes

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=np.sqrt(2.0))
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a2.data, gain=np.sqrt(2.0))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.downsample = nn.Conv1d(in_features, out_features, 1)

        self.bias = nn.Parameter(torch.zeros(num_nodes, out_features))

    def forward(self, input, adj):
        # Expect input: (B, N, in_features)
        print("GAT input shape:", input.shape)  # Debug print
        batch_size = input.size(0)

        h = torch.bmm(input, self.W.expand(batch_size, self.in_features, self.out_features)) # (B, N, out_features)
        print("After W multiplication h shape:", h.shape)  # Debug print

        f_1 = torch.bmm(h, self.a1.expand(batch_size, self.out_features, 1))  # (B, N, 1)
        f_2 = torch.bmm(h, self.a2.expand(batch_size, self.out_features, 1))  # (B, N, 1)
        e = self.leakyrelu(f_1 + f_2.transpose(2, 1))  # (B, N, N)
        print("Attention logits shape (e):", e.shape)

        attention = torch.mul(adj, e)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h) + self.bias.expand(batch_size, self.num_nodes, self.out_features)
        print("h_prime shape before residual adjustment:", h_prime.shape)

        if input.shape[-1] != h_prime.shape[-1]:
            input_ds = self.downsample(input.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
            print("Downsampled input shape:", input_ds.shape)
            h_prime = h_prime + input_ds
        else:
            h_prime = h_prime + input

        if self.concat:
            h_prime = F.elu(h_prime)
        print("GAT output shape:", h_prime.shape)

        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class STGATBlock(nn.Module):
    def __init__(self, cuda, in_channels, spatial_channels, out_channels,
                 num_nodes, num_timesteps_input, dropout=0.6, alpha=0.2, nheads=4, concat=True):
        super(STGATBlock, self).__init__()
        self.nheads = nheads
        self.concat = concat
        self.cuda = cuda
        self.spatial_channels = spatial_channels

        # TimeBlock transforms (B, F, N, T)
        self.temporal1 = nn.Sequential(TimeBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=2, nhid_channels=128, dropout=dropout, layer=3, cuda=cuda))

        # GAT in_features = out_channels * num_timesteps_input
        in_features = out_channels * num_timesteps_input
        self.attentions = [GraphAttentionLayer(in_features, spatial_channels, num_nodes=num_nodes,
                                               dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]

        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(num_nodes)

        if cuda:
            self.attentions = [att.cuda() for att in self.attentions]

    def forward(self, X, A_hat):
        # Initially, X: (B, N, F, T)
        print("STGATBlock input X shape:", X.shape)  # Debug print before TimeBlock

        # Permute to (B, F, N, T) for TimeBlock
        X = X.permute(0, 2, 1, 3)
        print("After permute for TimeBlock X shape:", X.shape)  # Debug print

        t = self.temporal1(X)  # (B, out_channels, N, T_in)
        print("After TimeBlock t shape:", t.shape)  # Debug print

        # Permute back to (B, N, F, T) for GAT
        t = t.permute(0, 2, 1, 3)
        print("Re-permute after TimeBlock t shape:", t.shape)  # Debug print

        B, N, outC, T_in = t.shape
        # Flatten features for GAT input: (B, N, outC*T_in)
        t = t.contiguous().view(B, N, outC * T_in)
        print("Flattened for GAT t shape:", t.shape)  # Debug print before GAT

        t_heads = [att(t, A_hat) for att in self.attentions]
        if self.concat:
            t2 = torch.cat(t_heads, dim=2)  # (B, N, nheads*spatial_channels)
        else:
            t2 = sum(t_heads) / self.nheads
        print("After all heads t2 shape:", t2.shape)  # Debug print after GAT heads

        # Reshape t2 -> (B, N, 1, spatial_channels)
        t2 = t2.view(B, N, 1, self.spatial_channels)
        print("Reshaped t2 shape:", t2.shape)  # Debug print before residual connection

        # Residual connection: Only add if dimensions match
        # Here, t2 has (B, N, 1, spatial_channels)
        # Original residual X was (B, N, F, T)
        # We skip adding residual since dimensions do not match
        out = self.relu(self.batch_norm(t2))
        print("STGATBlock output shape:", out.shape)  # Debug print final output of STGATBlock

        return out

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
        print('GatedLinearUnits input X shape:', X.shape)  # Debug print
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
            print("Downsample in GLU, new res shape:", res.shape)  # Debug print

        res = torch.mul(res, ones - gate)
        out = out + res
        out = self.relu(self.bn(out))
        print('GatedLinearUnits output X shape:', out.shape)  # Debug print
        return out

class EndConv(nn.Module):
    def __init__(self, in_channels, out_channels, nhid_channels, layer=3):
        super(EndConv, self).__init__()
        layers = []
        for i in range(layer):
            if i == 0:
                layers.append(GatedLinearUnits(in_channels, nhid_channels, kernel_size=1, dilation=1, cuda=True, groups=1))
            else:
                layers.append(GatedLinearUnits(nhid_channels, nhid_channels, kernel_size=3, dilation=1, cuda=True, groups=1))
        layers.append(nn.Conv1d(nhid_channels, out_channels, 1))
        self.units = nn.Sequential(*layers)
    
    def forward(self, X):
        print("EndConv input shape:", X.shape)  # Debug print
        out = self.units(X)
        print("EndConv output shape:", out.shape)  # Debug print
        return out

class STGAT(nn.Module):
    def __init__(self, cuda, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, nheads=4, nhid=64, layers=4, target_dim=6):
        super(STGAT, self).__init__()
        self.cuda_device = cuda
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.nheads = nheads
        self.nhid = nhid
        self.layers = layers
        self.target_dim = target_dim

        self.blocks = nn.ModuleList()
        for i in range(layers):
            in_channels = nhid
            n_input = nheads
            concat = True
            if i == 0:
                in_channels = num_features
                n_input = num_timesteps_input
            self.blocks.append(
                STGATBlock(
                    cuda=cuda,
                    in_channels=in_channels,
                    out_channels=nhid,
                    concat=concat,
                    spatial_channels=nhid,
                    num_nodes=num_nodes,
                    num_timesteps_input=n_input,
                    nheads=nheads
                )
            )

        self.output = EndConv(nhid, num_timesteps_output * target_dim, 512)
        self.time_decay_mult = nn.Parameter(torch.ones(1) * -0.1)

    def forward(self, A_hat, X, A_hat_=None, X_=None):
        # X: (B, N, F, T)
        print("STGAT input shape X:", X.shape)  # Debug print initial input

        out = X
        for i in range(self.layers):
            out = self.blocks[i](out, A_hat)  # (B, N, ?, nhid)

        B, N, F_out, T_in = out.shape
        print("Before EndConv, out shape:", out.shape)  # Debug print

        emb = out.reshape(B, N, F_out * T_in).permute(0, 2, 1)  # (B, F_out*T_in, N)
        print("Before EndConv, emb shape:", emb.shape)  # Debug print

        out4 = self.output(emb)  # (B, num_timesteps_output * target_dim, N)
        print("After EndConv output shape:", out4.shape)  # Debug print

        out4 = out4.permute(0, 2, 1)  # (B, N, T_out*target_dim)
        out4 = out4.view(B, N, self.num_timesteps_output, self.target_dim)  # (B, N, T_out, target_dim)
        print("STGAT output shape:", out4.shape)  # Debug print

        return out4
