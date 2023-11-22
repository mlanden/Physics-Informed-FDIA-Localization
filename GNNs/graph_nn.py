import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.datasets import KarateClub

dataset = KarateClub()

class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = gnn.GCNConv(dataset.num_features, 4)
        self.conv2 = gnn.GCNConv(4, 4)
        self.conv3 = gnn.GCNConv(4, 2)
        self.classifier = nn.Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)
        return out, h
    
model = GCN()
criterian = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(400):
    optim.zero_grad()
    out, h = model(dataset.x, dataset.edge_index)
    loss = criterian(out[dataset.train_mask], dataset.y[dataset.train_mask])
    loss.backward()
    optim.step()
    print(loss.item())