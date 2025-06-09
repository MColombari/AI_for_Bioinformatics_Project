import torch
from torch.nn import Linear, Sequential, ReLU, Dropout, Embedding, ModuleList
import torch.nn.functional as F
from torch_geometric import nn
from torch_geometric.nn import GCNConv, GATConv, NNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool


class EdgeAttrGNN(torch.nn.Module):
    def __init__(self, input_feature, hidden_channels, num_classes):
        super().__init__()
        self.nn1 = Sequential(
            Linear(1, 64),
            ReLU(),
            Linear(64, input_feature * hidden_channels)
        )
        self.conv1 = NNConv(input_feature, hidden_channels, self.nn1, aggr='mean')

        self.nn2 = Sequential(
            Linear(1, 64),
            ReLU(),
            Linear(64, hidden_channels * hidden_channels)
        )
        self.conv2 = NNConv(hidden_channels, hidden_channels, self.nn2, aggr='mean')

        self.dropout = Dropout(0.3)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

class EdgeAttrGNNLight(torch.nn.Module):
    def __init__(self, input_feature, hidden_channels, num_classes):
        super().__init__()
        self.nn1 = Sequential(
            Linear(1, 64),
            ReLU(),
            Linear(64, input_feature * hidden_channels)
        )
        self.conv1 = NNConv(input_feature, hidden_channels, self.nn1, aggr='mean')

        self.dropout = Dropout(0.3)
        self.lin1 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.lin1(x)
        return x

class EdgeAttrGAT(torch.nn.Module):
    def __init__(self, input_feature, hidden_channels, num_classes,
                 num_layers=3, heads=1, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # Embed edge attributes
        self.edge_emb = Linear(1, hidden_channels)

        # Initial GATv2Conv layer
        self.convs = ModuleList()
        self.convs.append(GATv2Conv(input_feature + hidden_channels, hidden_channels, heads=heads, concat=False))

        # Additional GATv2Conv layers
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False))

        # Output layer
        self.classifier = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        row, col = edge_index
        edge_attr_emb = self.edge_emb(edge_attr)  # [num_edges, hidden_channels]

        # Accumulate edge_attr_emb for each source node
        edge_attr_to_node = torch.zeros(x.size(0), edge_attr_emb.size(1), device=x.device)
        edge_attr_to_node.index_add_(0, row, edge_attr_emb)

        x = torch.cat([x, edge_attr_to_node], dim=1)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.classifier(x)


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
    

class SimpleGAT(torch.nn.Module):
    def __init__(self, input_feature:int, hidden_channels:list, num_head:int, num_classes:int, drop_out_prob:float):
        super(SimpleGAT, self).__init__()
        self.drop_out_prob = drop_out_prob

        self.conv1 = GATConv(input_feature, hidden_channels, heads=num_head, dropout=drop_out_prob)

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
    

class ComplexGAT(torch.nn.Module):
    def __init__(self, input_feature:int, hidden_channels:list, num_head:int, num_classes:int, drop_out_prob:float):
        super(ComplexGAT, self).__init__()
        self.drop_out_prob = drop_out_prob
        self.layer1 = GATConv(input_feature, hidden_channels, heads=num_head, dropout=drop_out_prob)
        self.layer2 = GATConv(hidden_channels * num_head, hidden_channels, heads=num_head, dropout=drop_out_prob)
        self.layer3 = GATConv(hidden_channels * num_head, hidden_channels, heads=num_head, dropout=drop_out_prob)
        self.layer4 = GATConv(hidden_channels * num_head, hidden_channels, heads=num_head, dropout=drop_out_prob)

        self.lin = nn.Linear(hidden_channels * num_head, num_classes)

    def forward(self, x, edge_index, batch=None):
        x = F.dropout(x, p=self.drop_out_prob, training=self.training)
        x = self.layer1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.drop_out_prob, training=self.training)
        x = self.layer2(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.drop_out_prob, training=self.training)
        x = self.layer3(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.drop_out_prob, training=self.training)
        x = self.layer4(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)

        if torch.isnan(x).any():
            raise Exception("NaN detected in linear layer")
        
        x = F.log_softmax(x, dim=1)

        return x