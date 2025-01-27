import torch
from Save_model import SaveModel as SM

# So we have a structure of folder where we have a main folder containig all the test for
# each subgroup (Methylation, Gene, Copy number), and in each of these folder we have
# a folder for each test, inside all the results and checkpoint are saved.

#   Data Parameter

# Name of the test, like methylation or gene... .
TEST_NAME = "Test name"
MORE_INFO = """
    More information.
"""

# PATH where we'll create the folder containig the new test.
TEST_FOLDER_PATH = "."

# Load previous checkpoint.
START_FROM_CHECKPOINT = False
CHECKPOINT_PATH = "."

#   Model parameter
hyperparameter = {
    'num_classes': 2,
    'epochs': 500,
    'batch_size': 10
}

model_structure_description = """
    How many layer, and how are they made
    Just copy and paste what's inside the model
"""

# Create Folder and first files.
sm = SM(TEST_FOLDER_PATH, TEST_NAME)
sm.save_test_info(MORE_INFO, START_FROM_CHECKPOINT, CHECKPOINT_PATH)
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
sm.save_model_hyperparameter(hyperparameter, model_structure_description)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create Folder

#from torch.nn import Linear
#import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
#from torch_geometric.nn import global_mean_pool
#
#class GCN(torch.nn.Module):
#    def __init__(self, hidden_channels):
#        super(GCN, self).__init__()
#        torch.manual_seed(12345)
#        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
#        self.conv2 = GCNConv(hidden_channels, hidden_channels)
#        self.conv3 = GCNConv(hidden_channels, hidden_channels)
#        self.lin = Linear(hidden_channels, dataset.num_classes)
#
#    def forward(self, x, edge_index, batch):
#        # 1. Obtain node embeddings 
#        x = self.conv1(x, edge_index)
#        x = x.relu()
#        x = self.conv2(x, edge_index)
#        x = x.relu()
#        x = self.conv3(x, edge_index)
#
#        # 2. Readout layer
#        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#
#        # 3. Apply a final classifier
#        x = F.dropout(x, p=0.5, training=self.training)
#        x = self.lin(x)
#        
#        return x
#
#model = GCN(hidden_channels=64)
#print(model)