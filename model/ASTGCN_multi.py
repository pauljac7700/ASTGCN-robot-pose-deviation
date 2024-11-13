# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial

# ---------------------------------------------------------------------
# Spatial Attention Layer
# ---------------------------------------------------------------------

class Spatial_Attention_layer(nn.Module):
    '''
    Compute spatial attention scores.

    This layer calculates spatial attention scores for the input graph data,
    capturing the dynamic correlations between different nodes over time.
    '''

    def __init__(self, DEVICE: torch.device, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        '''
        Initialize the spatial attention layer.

        :param DEVICE: The device (CPU or GPU) to run computations on.
        :param in_channels: Number of input feature channels.
        :param num_of_vertices: Number of nodes in the graph.
        :param num_of_timesteps: Number of time steps in the input sequence.
        '''
        super(Spatial_Attention_layer, self).__init__()
        # Initialize learnable parameters for spatial attention
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))  # Shape: (T,)
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))  # Shape: (F_in, T)
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))  # Shape: (F_in,)
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))  # Shape: (1, N, N)
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))  # Shape: (N, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the spatial attention layer.

        :param x: Input tensor of shape (batch_size, N, F_in, T)
        :return: Spatial attention scores of shape (batch_size, N, N)
        '''
        # x shape: (B, N, F_in, T)
        batch_size, N, F_in, T = x.shape

        # Compute lhs: x multiplied by W1 and W2
        # x_W1: (B, N, F_in, T) * (T,) -> (B, N, F_in, T)
        x_W1 = x * self.W1  # Broadcasting W1 over the last dimension

        # Sum over time dimension T to get (B, N, F_in)
        x_W1_sum = x_W1.sum(dim=3)  # Shape: (B, N, F_in)

        # Multiply x_W1_sum with W2: (B, N, F_in) @ (F_in, T) -> (B, N, T)
        lhs = torch.matmul(x_W1_sum, self.W2)  # Shape: (B, N, T)

        # Compute rhs: W3 * x, then sum over feature dimension F_in
        # W3: (F_in,)
        x_W3 = x * self.W3.view(1, 1, -1, 1)  # Shape: (B, N, F_in, T)
        x_W3_sum = x_W3.sum(dim=2)  # Sum over F_in, shape: (B, N, T)

        # Transpose x_W3_sum to get rhs: (B, T, N)
        rhs = x_W3_sum.permute(0, 2, 1)  # Shape: (B, T, N)

        # Compute product: lhs (B, N, T) @ rhs (B, T, N) -> (B, N, N)
        product = torch.matmul(lhs, rhs)  # Shape: (B, N, N)

        # Add bias and apply activation function
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # Shape: (B, N, N)

        # Normalize the attention scores with softmax over nodes
        S_normalized = F.softmax(S, dim=-1)  # Softmax over the last dimension (nodes)

        return S_normalized  # Shape: (B, N, N)

# ---------------------------------------------------------------------
# Temporal Attention Layer
# ---------------------------------------------------------------------

class Temporal_Attention_layer(nn.Module):
    '''
    Compute temporal attention scores.

    This layer calculates temporal attention scores for the input graph data,
    capturing the dynamic correlations over different time steps.
    '''

    def __init__(self, DEVICE: torch.device, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        '''
        Initialize the temporal attention layer.

        :param DEVICE: The device (CPU or GPU) to run computations on.
        :param in_channels: Number of input feature channels.
        :param num_of_vertices: Number of nodes in the graph.
        :param num_of_timesteps: Number of time steps in the input sequence.
        '''
        super(Temporal_Attention_layer, self).__init__()
        # Initialize learnable parameters for temporal attention
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))  # Shape: (N,)
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))  # Shape: (F_in, N)
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))  # Shape: (F_in,)
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))  # Shape: (1, T, T)
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))  # Shape: (T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the temporal attention layer.

        :param x: Input tensor of shape (batch_size, N, F_in, T)
        :return: Temporal attention scores of shape (batch_size, T, T)
        '''
        # x shape: (B, N, F_in, T)
        batch_size, N, F_in, T = x.shape

        # Permute x to shape: (B, T, F_in, N)
        x_permuted = x.permute(0, 3, 2, 1)  # Shape: (B, T, F_in, N)

        # Compute lhs
        x_U1 = x_permuted * self.U1  # (B, T, F_in, N) * (N,) -> (B, T, F_in, N)
        x_U1_sum = x_U1.sum(dim=3)  # Sum over N, shape: (B, T, F_in)
        lhs = torch.matmul(x_U1_sum, self.U2)  # (B, T, F_in) @ (F_in, N) -> (B, T, N)

        # Compute rhs
        x_U3 = x * self.U3.view(1, 1, -1, 1)  # (B, N, F_in, T)
        x_U3_sum = x_U3.sum(dim=2)  # Sum over F_in, shape: (B, N, T)
        rhs = x_U3_sum.permute(0, 2, 1)  # (B, T, N)

        # Compute product
        product = torch.matmul(lhs, rhs.transpose(-1, -2))  # (B, T, N) @ (B, N, T) -> (B, T, T)

        # Compute temporal attention scores
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        # Normalize the attention scores with softmax over time steps
        E_normalized = F.softmax(E, dim=-1)  # Softmax over the last dimension (time steps)

        return E_normalized  # Shape: (B, T, T)

# ---------------------------------------------------------------------
# Chebyshev Graph Convolution with Spatial Attention
# ---------------------------------------------------------------------

class cheb_conv_withSAt(nn.Module):
    '''
    K-order Chebyshev graph convolution with spatial attention.

    This layer applies Chebyshev graph convolution operations incorporating spatial attention scores.
    '''

    def __init__(self, K: int, cheb_polynomials: list, in_channels: int, out_channels: int):
        '''
        Initialize the Chebyshev convolution layer with spatial attention.

        :param K: Order of the Chebyshev polynomials.
        :param cheb_polynomials: List of Chebyshev polynomials (as torch tensors).
        :param in_channels: Number of input feature channels.
        :param out_channels: Number of output feature channels.
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials  # List of tensors, each of shape (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device

        # Initialize learnable parameters Theta_k for each Chebyshev order k
        self.Theta = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))
            for _ in range(K)
        ])

    def forward(self, x: torch.Tensor, spatial_attention: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for Chebyshev convolution with spatial attention.

        :param x: Input tensor of shape (batch_size, N, F_in, T)
        :param spatial_attention: Spatial attention tensor of shape (batch_size, N, N)
        :return: Output tensor of shape (batch_size, N, F_out, T)
        '''
        batch_size, N, F_in, T = x.shape

        outputs = []

        for t in range(T):
            # Get graph signal at time step t: (batch_size, N, F_in)
            graph_signal = x[:, :, :, t]  # Shape: (B, N, F_in)

            # Initialize output for this time step
            output = torch.zeros(batch_size, N, self.out_channels).to(self.DEVICE)  # Shape: (B, N, F_out)

            for k in range(self.K):
                # Chebyshev polynomial T_k: Shape (N, N)
                T_k = self.cheb_polynomials[k]  # Shape: (N, N)

                # Apply spatial attention
                # Expand T_k to batch size
                T_k = T_k.unsqueeze(0).expand(batch_size, N, N)  # Shape: (B, N, N)
                T_k_with_at = T_k * spatial_attention  # Element-wise multiplication

                # Theta_k: Shape (F_in, F_out)
                theta_k = self.Theta[k]  # Shape: (F_in, F_out)

                # Compute convolution
                rhs = torch.bmm(T_k_with_at, graph_signal)  # (B, N, N) @ (B, N, F_in) -> (B, N, F_in)

                output += torch.matmul(rhs, theta_k)  # (B, N, F_in) @ (F_in, F_out) -> (B, N, F_out)

            outputs.append(output.unsqueeze(-1))  # Shape: (B, N, F_out, 1)

        # Concatenate outputs over time dimension
        outputs = torch.cat(outputs, dim=-1)  # Shape: (B, N, F_out, T)

        return F.relu(outputs)  # Apply ReLU activation

# ---------------------------------------------------------------------
# ASTGCN Block
# ---------------------------------------------------------------------

class ASTGCN_block(nn.Module):
    '''
    ASTGCN block consisting of temporal attention, spatial attention,
    Chebyshev graph convolution, and time convolution layers.
    '''

    def __init__(self, DEVICE: torch.device, in_channels: int, K: int, nb_chev_filter: int, nb_time_filter: int,
                 time_strides: int, cheb_polynomials: list, num_of_vertices: int, num_of_timesteps: int):
        '''
        Initialize the ASTGCN block.

        :param DEVICE: The device to run computations on.
        :param in_channels: Number of input feature channels.
        :param K: Order of Chebyshev polynomials.
        :param nb_chev_filter: Number of Chebyshev filters.
        :param nb_time_filter: Number of time filters.
        :param time_strides: Strides for time convolution.
        :param cheb_polynomials: List of Chebyshev polynomials.
        :param num_of_vertices: Number of nodes in the graph.
        :param num_of_timesteps: Number of time steps in the input sequence.
        '''
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(
            in_channels=nb_chev_filter,
            out_channels=nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1)
        )
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=nb_time_filter,
            kernel_size=(1, 1),
            stride=(1, time_strides)
        )
        self.ln = nn.LayerNorm(nb_time_filter)  # Layer normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the ASTGCN block.

        :param x: Input tensor of shape (batch_size, N, F_in, T_in)
        :return: Output tensor of shape (batch_size, N, nb_time_filter, T_out)
        '''
        batch_size, N, F_in, T_in = x.shape

        # Temporal Attention
        temporal_At = self.TAt(x)  # Shape: (B, T_in, T_in)

        # Apply temporal attention
        x_TAt = torch.matmul(x.reshape(batch_size, -1, T_in), temporal_At)  # Shape: (B, N * F_in, T_in)
        x_TAt = x_TAt.reshape(batch_size, N, F_in, T_in)  # Shape: (B, N, F_in, T_in)

        # Spatial Attention
        spatial_At = self.SAt(x_TAt)  # Shape: (B, N, N)

        # Chebyshev Graph Convolution with Spatial Attention
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # Shape: (B, N, nb_chev_filter, T_in)

        # Temporal Convolution
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # Shape: (B, nb_time_filter, N, T_out)

        # Residual Shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # Shape: (B, nb_time_filter, N, T_out)

        # Add and apply activation
        output = F.relu(x_residual + time_conv_output)  # Shape: (B, nb_time_filter, N, T_out)

        # Apply layer normalization
        output = output.permute(0, 2, 3, 1)  # Shape: (B, N, T_out, nb_time_filter)
        output = self.ln(output)  # LayerNorm over the last dimension
        output = output.permute(0, 1, 3, 2)  # Shape: (B, N, nb_time_filter, T_out)

        return output  # Shape: (B, N, nb_time_filter, T_out)

# ---------------------------------------------------------------------
# ASTGCN Submodule
# ---------------------------------------------------------------------

class ASTGCN_submodule(nn.Module):
    '''
    ASTGCN submodule for spatial-temporal graph convolution.

    This module stacks multiple ASTGCN blocks and applies a final convolution
    to produce the desired output with multiple target dimensions.
    '''

    def __init__(self, DEVICE: torch.device, nb_block: int, in_channels: int, K: int, nb_chev_filter: int,
                 nb_time_filter: int, time_strides: int, cheb_polynomials: list, num_for_predict: int,
                 len_input: int, num_of_vertices: int, target_dim: int):
        '''
        Initialize the ASTGCN submodule.

        :param DEVICE: The device to run computations on.
        :param nb_block: Number of ASTGCN blocks.
        :param in_channels: Number of input feature channels.
        :param K: Order of Chebyshev polynomials.
        :param nb_chev_filter: Number of Chebyshev filters.
        :param nb_time_filter: Number of time filters.
        :param time_strides: Strides for time convolution.
        :param cheb_polynomials: List of Chebyshev polynomials.
        :param num_for_predict: Number of future time steps to predict.
        :param len_input: Length of input sequence.
        :param num_of_vertices: Number of nodes in the graph.
        :param target_dim: Number of target dimensions (e.g., pose error components).
        '''
        super(ASTGCN_submodule, self).__init__()

        self.DEVICE = DEVICE
        self.num_for_predict = num_for_predict
        self.target_dim = target_dim

        # Create ASTGCN blocks
        self.BlockList = nn.ModuleList()
        self.BlockList.append(ASTGCN_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                                           time_strides, cheb_polynomials, num_of_vertices, len_input))
        for _ in range(nb_block - 1):
            self.BlockList.append(ASTGCN_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter,
                                               1, cheb_polynomials, num_of_vertices, len_input // time_strides))

        # Final convolution layer to output num_for_predict * target_dim features
        self.final_conv = nn.Conv2d(
            in_channels=int(len_input / time_strides),
            out_channels=num_for_predict * target_dim,
            kernel_size=(1, nb_time_filter)
        )

        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the ASTGCN submodule.

        :param x: Input tensor of shape (batch_size, N, F_in, T_in)
        :return: Output tensor of shape (batch_size, N, num_for_predict, target_dim)
        '''
        # x shape: (B, N, F_in, T_in)
        for block in self.BlockList:
            x = block(x)  # Output shape after each block: (B, N, nb_time_filter, T_out)

        # x shape after blocks: (B, N, nb_time_filter, T_out)
        # Permute dimensions to (B, T_out, N, nb_time_filter)
        x = x.permute(0, 3, 1, 2)  # Shape: (B, T_out, N, nb_time_filter)

        # Apply final convolution
        x = self.final_conv(x)  # Shape: (B, num_for_predict * target_dim, N, 1)

        # Remove last dimension
        x = x.squeeze(-1)  # Shape: (B, num_for_predict * target_dim, N)

        # Permute to (B, N, num_for_predict * target_dim)
        x = x.permute(0, 2, 1)  # Shape: (B, N, num_for_predict * target_dim)

        # Reshape to separate num_for_predict and target_dim
        x = x.view(x.shape[0], x.shape[1], self.num_for_predict, self.target_dim)  # Shape: (B, N, num_for_predict, target_dim)

        return x  # Output shape: (B, N, num_for_predict, target_dim)

# ---------------------------------------------------------------------
# Model Creation Function
# ---------------------------------------------------------------------

def make_model(DEVICE: torch.device, nb_block: int, in_channels: int, K: int, nb_chev_filter: int,
               nb_time_filter: int, time_strides: int, adj_mx: torch.Tensor, num_for_predict: int,
               len_input: int, num_of_vertices: int, target_dim: int) -> nn.Module:
    '''
    Create the ASTGCN model.

    :param DEVICE: The device to run computations on.
    :param nb_block: Number of ASTGCN blocks.
    :param in_channels: Number of input feature channels.
    :param K: Order of Chebyshev polynomials.
    :param nb_chev_filter: Number of Chebyshev filters.
    :param nb_time_filter: Number of time filters.
    :param time_strides: Time strides for temporal convolution.
    :param adj_mx: Adjacency matrix of the graph (numpy.ndarray or torch.Tensor).
    :param num_for_predict: Number of future time steps to predict.
    :param len_input: Length of input sequence.
    :param num_of_vertices: Number of nodes in the graph.
    :param target_dim: Number of target dimensions.
    :return: ASTGCN model.
    '''
    # Compute scaled Laplacian and Chebyshev polynomials
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = cheb_polynomial(L_tilde, K)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomials]

    # Create the model
    model = ASTGCN_submodule(
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
        target_dim=target_dim
    )

    # Initialize model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
