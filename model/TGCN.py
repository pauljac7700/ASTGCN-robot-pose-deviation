# TGCN.py

import torch
import torch.nn as nn
from lib.TGCN_graph_conv import calculate_laplacian_with_self_loop


class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, in_channels: int, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._in_channels = in_channels
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer("laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self.weights = nn.Parameter(torch.FloatTensor(self._in_channels + self._num_gru_units, self._output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes, in_channels = inputs.shape
        hidden_state = hidden_state.reshape(batch_size, num_nodes, self._num_gru_units)
        concatenation = torch.cat((inputs, hidden_state), dim=2)  # (batch_size, num_nodes, in_channels + num_gru_units)
        # Graph convolution
        a_times_concat = torch.matmul(self.laplacian, concatenation)  # (batch_size, num_nodes, in_channels + num_gru_units)
        # Linear transformation
        outputs = torch.matmul(a_times_concat, self.weights) + self.biases  # (batch_size, num_nodes, output_dim)
        return outputs


class TGCNCell(nn.Module):
    def __init__(self, adj, in_channels: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._in_channels = in_channels
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._in_channels, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._in_channels, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # Compute [r, u]
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))  # (batch_size, num_nodes, 2 * hidden_dim)
        r, u = torch.chunk(concatenation, chunks=2, dim=2)
        # Compute candidate hidden state c
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # Update hidden state
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state


class TGCN(nn.Module):
    def __init__(self, adj, in_channels: int, hidden_dim: int):
        super(TGCN, self).__init__()
        self._num_nodes = adj.shape[0]
        self._in_channels = in_channels
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._in_channels, self._hidden_dim)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, in_channels = inputs.shape
        assert self._num_nodes == num_nodes
        assert self._in_channels == in_channels
        hidden_state = torch.zeros(batch_size, num_nodes, self._hidden_dim).type_as(inputs)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :, :], hidden_state)
        return output  # (batch_size, num_nodes, hidden_dim)


class TGCNWithGlobalOutput(nn.Module):
    def __init__(self, adj, in_channels, hidden_dim):
        super(TGCNWithGlobalOutput, self).__init__()
        self.tgcn = TGCN(adj, in_channels, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 6)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, num_nodes, in_channels)
        output = self.tgcn(inputs)  # (batch_size, num_nodes, hidden_dim)
        # Aggregate node features
        aggregated_output = output.mean(dim=1)  # (batch_size, hidden_dim)
        final_output = self.output_layer(aggregated_output)  # (batch_size, 6)
        return final_output
