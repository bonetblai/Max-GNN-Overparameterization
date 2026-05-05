import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class MaxGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MaxGNNLayer, self).__init__(aggr='max')
        self.lin = nn.Linear(in_channels, out_channels)
        self.update_lin = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        msg = self.propagate(edge_index, x=x)
        combined = torch.cat([x, msg], dim=1)
        return F.relu(self.update_lin(combined))

    def message(self, x_j):
        # Apply transformation BEFORE aggregation
        return self.lin(x_j)

    def update(self, aggr_out):
        # Handled in forward for more flexibility with skip connections/concatenation
        return aggr_out

class MaxGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(MaxGNN, self).__init__()
        self.layers = nn.ModuleList()
        # Initial layer
        self.layers.append(MaxGNNLayer(in_channels, hidden_channels))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(MaxGNNLayer(hidden_channels, hidden_channels))
        # Output layer
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.classifier(x)
