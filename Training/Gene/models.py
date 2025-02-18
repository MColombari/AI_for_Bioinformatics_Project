import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class simple_GCN(torch.nn.Module):
    def __init__(self, input_feature, hidden_channels, num_classes):
        # input feature comes from dataset.num_node_feature
        # which returns the number of features per node in the dataset.
        # So input_feature need to be number of feature per node in the dataset.
        # source: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html
        super(simple_GCN, self).__init__()
        self.conv1 = GCNConv(input_feature, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None):
        # num_nodes = x.shape[0]  # Numero di nodi
        # print(x.shape)
        # max_index = edge_index.max().item()
        # print(f"Numero di nodi: {num_nodes}, Max edge_index: {max_index}")
        # assert max_index < num_nodes, "Errore: edge_index contiene un indice fuori dai limiti!"
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x 
    

class small_GCN(torch.nn.Module):
    def __init__(self, input_feature, hidden_channels, num_classes):
        # input feature comes from dataset.num_node_feature
        # which returns the number of features per node in the dataset.
        # So input_feature need to be number of feature per node in the dataset.
        # source: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html
        super(small_GCN, self).__init__()
        self.conv1 = GCNConv(input_feature, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None):
        # num_nodes = x.shape[0]  # Numero di nodi
        # print(x.shape)
        # max_index = edge_index.max().item()
        # print(f"Numero di nodi: {num_nodes}, Max edge_index: {max_index}")
        # assert max_index < num_nodes, "Errore: edge_index contiene un indice fuori dai limiti!"
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x 