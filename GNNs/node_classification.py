import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root="../../data/graph/Planetoid", name="Cora", transform=NormalizeFeatures())

class GCN(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.conv1 = gnn.GCNConv(dataset.num_features, hidden)
        self.conv2 = gnn.GCNConv(hidden, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
model = GCN(16)

optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterian = nn.CrossEntropyLoss()

for epoch in range(100):
    optim.zero_grad()
    out = model(dataset.x, dataset.edge_index)
    loss = criterian(out[dataset.train_mask], dataset.y[dataset.train_mask])
    loss.backward()
    optim.step()
    print(f"Epoch: {epoch}, loss: {loss.item()}")