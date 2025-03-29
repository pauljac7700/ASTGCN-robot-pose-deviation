# ASTGCN_no_temporal.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial

# Import original classes but we won't use temporal attention

class Spatial_Attention_layer(nn.Module):
    # Same as original Spatial_Attention_layer
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        batch_size, N, F_in, T = x.shape
        x_W1 = x * self.W1
        x_W1_sum = x_W1.sum(dim=3)
        lhs = torch.matmul(x_W1_sum, self.W2)
        x_W3 = x * self.W3.view(1,1,-1,1)
        x_W3_sum = x_W3.sum(dim=2)
        rhs = x_W3_sum.permute(0,2,1)
        product = torch.matmul(lhs, rhs)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        S_normalized = F.softmax(S, dim=-1)
        return S_normalized

class cheb_conv_withSAt(nn.Module):
    # Same as original cheb_conv_withSAt
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))
            for _ in range(K)
        ])

    def forward(self, x, spatial_attention):
        batch_size, N, F_in, T = x.shape
        outputs = []
        for t in range(T):
            graph_signal = x[:,:,:,t]
            output = torch.zeros(batch_size, N, self.out_channels).to(self.DEVICE)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                T_k = T_k.unsqueeze(0).expand(batch_size, N, N)
                T_k_with_at = T_k * spatial_attention
                theta_k = self.Theta[k]
                rhs = torch.bmm(T_k_with_at, graph_signal)
                output += torch.matmul(rhs, theta_k)
            outputs.append(output.unsqueeze(-1))
        outputs = torch.cat(outputs, dim=-1)
        return F.relu(outputs)

class ASTGCN_block_no_temporal(nn.Module):
    '''
    ASTGCN block without temporal attention.
    We keep spatial attention and cheb_conv_withSAt as is.
    '''
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(ASTGCN_block_no_temporal, self).__init__()
        # No Temporal_Attention_layer
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(
            in_channels=nb_chev_filter,
            out_channels=nb_time_filter,
            kernel_size=(1,3),
            stride=(1,time_strides),
            padding=(0,1)
        )
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=nb_time_filter,
            kernel_size=(1,1),
            stride=(1,time_strides)
        )
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        # Without temporal attention, just pass x directly
        spatial_At = self.SAt(x)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)

        time_conv_output = self.time_conv(spatial_gcn.permute(0,2,1,3))
        x_residual = self.residual_conv(x.permute(0,2,1,3))
        output = F.relu(x_residual + time_conv_output)
        output = output.permute(0,2,3,1)
        output = self.ln(output)
        output = output.permute(0,1,3,2)
        return output

class ASTGCN_submodule_no_temporal(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices, residual_dim):
        super(ASTGCN_submodule_no_temporal, self).__init__()
        self.DEVICE = DEVICE
        self.num_for_predict = num_for_predict
        self.residual_dim = residual_dim

        self.BlockList = nn.ModuleList()
        self.BlockList.append(ASTGCN_block_no_temporal(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                                                       time_strides, cheb_polynomials, num_of_vertices, len_input))
        for _ in range(nb_block - 1):
            self.BlockList.append(ASTGCN_block_no_temporal(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter,
                                                           1, cheb_polynomials, num_of_vertices, len_input // time_strides))

        self.final_conv = nn.Conv2d(
            in_channels=int(len_input/time_strides),
            out_channels=num_for_predict * residual_dim,
            kernel_size=(1, nb_time_filter)
        )

        self.to(DEVICE)

    def forward(self, x):
        for block in self.BlockList:
            x = block(x)
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)
        x = x.squeeze(-1)
        x = x.permute(0,2,1)
        x = x.view(x.shape[0], x.shape[1], self.num_for_predict, self.residual_dim)
        return x

def make_model_no_temporal(DEVICE, nb_block, in_channels, K, nb_chev_filter,
                           nb_time_filter, time_strides, adj_mx, num_for_predict,
                           len_input, num_of_vertices, residual_dim):
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = cheb_polynomial(L_tilde, K)
    cheb_polynomials = [torch.from_numpy(i).float().to(DEVICE) for i in cheb_polynomials]

    model = ASTGCN_submodule_no_temporal(
        DEVICE=DEVICE,
        nb_block=nb_block,
        in_channels=in_channels,
        K=K,
        nb_chev_filter=nb_chev_filter,
        nb_time_filter=nb_time_filter,
        time_strides=time_strides,
        cheb_polynomials=cheb_polynomials,
        num_for_predict=num_for_predict,
        len_input=len_input,
        num_of_vertices=num_of_vertices,
        residual_dim=residual_dim
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
