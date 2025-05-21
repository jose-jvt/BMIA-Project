import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4, dropout=0.2):
        super(GATRegressor, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=True, dropout=dropout)

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, 1)  # Output is a single float (BMI)

    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x.view(-1)
