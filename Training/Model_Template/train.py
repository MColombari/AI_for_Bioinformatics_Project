import torch
from Save_model import SaveModel as SM
from models import simple_GCN
from Load_and_Process_Data import LPD
from torch_geometric.loader import DataLoader

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

#   Model parameter TODO
hyperparameter = {
    'num_classes': 2,
    'epochs': 500,
    'batch_size': 10,
    'seed': 123456
}

torch.manual_seed(hyperparameter['seed'])

# Create Folder and first files.
sm = SM(TEST_FOLDER_PATH, TEST_NAME)
sm.save_test_info(MORE_INFO, START_FROM_CHECKPOINT, CHECKPOINT_PATH)
sm.save_model_hyperparameter(hyperparameter)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

node_feature_number = None #TODO
model = simple_GCN(node_feature_number, 10, hyperparameter['num_classes'])
                                       #TODO

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
sm.save_model_architecture(model)

# https://pytorch-geometric.readthedocs.io/en/2.5.3/notes/create_dataset.html
lpd = LPD()
data_list = lpd.get_data()  # List of Data.
loader = DataLoader(data_list, batch_size=hyperparameter['batch_size'])