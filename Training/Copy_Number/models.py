import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric import nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool

class simple_GCN(torch.nn.Module):
    def __init__(self, input_feature, num_classes):
        # input feature comes from dataset.num_node_feature
        # which returns the number of features per node in the dataset.
        # So input_feature need to be number of feature per node in the dataset.
        # source: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html
        super(simple_GCN, self).__init__()
        self.conv1 = GCNConv(input_feature, 1000)
        self.conv2 = GCNConv(1000, 700)
        self.conv3 = GCNConv(700, 200)
        self.lin = Linear(200, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch=None):
        # print(x.device)
        # num_nodes = x.shape[0]  # Numero di nodi
        # print(x.shape)
        # max_index = edge_index.max().item()
        # print(f"Numero di nodi: {num_nodes}, Max edge_index: {max_index}")
        # assert max_index < num_nodes, "Errore: edge_index contiene un indice fuori dai limiti!"
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
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
    


class GAT(torch.nn.Module):
    def __init__(self, input_feature:int, hidden_channels:list, num_head:int, num_classes:int, drop_out_prob:float):
        super(GAT, self).__init__()
        self.drop_out_prob = drop_out_prob

        self.conv1 = GATConv(input_feature, hidden_channels, heads=num_head, dropout=drop_out_prob)
        self.conv2 = GATConv(hidden_channels * num_head, hidden_channels, heads=num_head, dropout=drop_out_prob)

        self.lin = nn.Linear(hidden_channels * num_head, num_classes)

    def forward(self, x, edge_index, batch=None):

        # x = x.type(torch.float)

        x = F.dropout(x, p=self.drop_out_prob, training=self.training)
        # print(x)
        if torch.isnan(x).any():
            raise Exception("NaN detected before first conv")
        x = self.conv1(x, edge_index)
        # print(x)
        if torch.isnan(x).any():
            raise Exception("NaN detected in first conv")
        x = F.elu(x)
        x = F.dropout(x, p=self.drop_out_prob, training=self.training)
        x = self.conv2(x, edge_index)
        # print(x)
        if torch.isnan(x).any():
            raise Exception("NaN detected in second conv")
        x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)

        # print(x)
        if torch.isnan(x).any():
            raise Exception("NaN detected in linear layer")
        
        x = F.log_softmax(x, dim=1)

        # print(x)
        if torch.isnan(x).any():
            raise Exception("NaN detected in output model")

        return x