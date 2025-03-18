# ASTGCN_no_spatial.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial
# We'll reuse Temporal_Attention_layer from original, but remove Spatial_Attention_layer usage

class Temporal_Attention_layer(nn.Module):
    # Same as original
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        B, N, F_in, T = x.shape
        x_permuted = x.permute(0,3,2,1)
        x_U1 = x_permuted * self.U1
        x_U1_sum = x_U1.sum(dim=3)
        lhs = torch.matmul(x_U1_sum, self.U2)
        x_U3 = x * self.U3.view(1,1,-1,1)
        x_U3_sum = x_U3.sum(dim=2)
        rhs = x_U3_sum.permute(0,2,1)
        product = torch.matmul(lhs, rhs.transpose(-1,-2))
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        E_normalized = F.softmax(E, dim=-1)
        return E_normalized

class ASTGCN_block_no_spatial(nn.Module):
    '''
    ASTGCN block without spatial attention.
    We keep temporal attention, but cheb_conv will be simplified (no spatial_at).
    '''
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(ASTGCN_block_no_spatial, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.nb_chev_filter = nb_chev_filter

        # Theta parameters for Chebyshev convolution without spatial attention
        self.Theta = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, nb_chev_filter).to(DEVICE))
            for _ in range(K)
        ])

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
        B, N, F_in, T = x.shape
        temporal_At = self.TAt(x)
        x_TAt = torch.matmul(x.reshape(B,-1,T), temporal_At).reshape(B,N,F_in,T)

        # Chebyshev convolution without spatial attention
        outputs = []
        for t in range(T):
            graph_signal = x_TAt[:,:,:,t]
            out_t = torch.zeros(B, N, self.nb_chev_filter, device=x.device)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                rhs = torch.matmul(T_k, graph_signal) # (N,N)(B,N,F_in)->(B,N,F_in)
                out_t += torch.matmul(rhs, self.Theta[k]) 
            outputs.append(F.relu(out_t).unsqueeze(-1))
        spatial_gcn = torch.cat(outputs, dim=-1) # (B,N,nb_chev_filter,T)

        time_conv_output = self.time_conv(spatial_gcn.permute(0,2,1,3))
        x_residual = self.residual_conv(x.permute(0,2,1,3))
        output = F.relu(x_residual + time_conv_output)
        output = output.permute(0,2,3,1)
        output = self.ln(output)
        output = output.permute(0,1,3,2)
        return output

class ASTGCN_submodule_no_spatial(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices, residual_dim):
        super(ASTGCN_submodule_no_spatial, self).__init__()
        self.DEVICE = DEVICE
        self.num_for_predict = num_for_predict
        self.residual_dim = residual_dim

        self.BlockList = nn.ModuleList()
        self.BlockList.append(ASTGCN_block_no_spatial(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                                                      time_strides, cheb_polynomials, num_of_vertices, len_input))
        for _ in range(nb_block - 1):
            self.BlockList.append(ASTGCN_block_no_spatial(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter,
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

def make_model_no_spatial(DEVICE, nb_block, in_channels, K, nb_chev_filter,
                          nb_time_filter, time_strides, adj_mx, num_for_predict,
                          len_input, num_of_vertices, residual_dim):
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = cheb_polynomial(L_tilde, K)
    cheb_polynomials = [torch.from_numpy(i).float().to(DEVICE) for i in cheb_polynomials]

    model = ASTGCN_submodule_no_spatial(
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
