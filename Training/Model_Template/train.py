import torch
from Save_model import SaveModel as SM
from models import simple_GCN
from Load_and_Process_Data import LPD
from torch_geometric.loader import DataLoader
from collections import OrderedDict

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
    'seed': 123456,
    'num_workers': 12
}

torch.manual_seed(hyperparameter['seed'])

# Create Folder and first files.
sm = SM(TEST_FOLDER_PATH, TEST_NAME)
sm.save_test_info(MORE_INFO, START_FROM_CHECKPOINT, CHECKPOINT_PATH)
sm.save_model_hyperparameter(hyperparameter)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://pytorch-geometric.readthedocs.io/en/2.5.3/notes/create_dataset.html
lpd = LPD()
data_train_list, data_test_list = lpd.get_data()  # List of Data.
# Inside of data we need to specify which y we have.

train = DataLoader(data_train_list, batch_size=hyperparameter['batch_size'], shuffle=True, num_workers=hyperparameter['num_workers'], pin_memory=True)
test = DataLoader(data_test_list, batch_size=hyperparameter['batch_size'], shuffle=True, num_workers=hyperparameter['num_workers'], pin_memory=True)
# pin_memory=True will automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.
# https://pytorch.org/docs/stable/data.html.


node_feature_number = None #TODO
model = simple_GCN(node_feature_number, 10, hyperparameter['num_classes'])
                                       #TODO

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
sm.save_model_architecture(model)

if START_FROM_CHECKPOINT:
    checkpoint = torch.load(CHECKPOINT_PATH)
    s_epoch = checkpoint['epoch']
    model_dict = checkpoint['model_dict']
    new_state_dict = OrderedDict()
    for k, v in model_dict.items():
        name = k[7:] # remove 'module.'
        new_state_dict[name]=v
    model.load_state_dict(new_state_dict)