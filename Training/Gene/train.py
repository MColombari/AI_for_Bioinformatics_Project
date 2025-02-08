import torch
from Save_model import SaveModel as SM
from models import simple_GCN
from Load_and_Process_Data import LPD
from torch_geometric.loader import DataLoader
from collections import OrderedDict
from sklearn.metrics import accuracy_score
import numpy as np

# So we have a structure of folder where we have a main folder containig all the test for
# each subgroup (Methylation, Gene, Copy number), and in each of these folder we have
# a folder for each test, inside all the results and checkpoint are saved.

#   Data Parameter

# Name of the test, like methylation or gene... .
TEST_NAME = "Train_Gene"
MORE_INFO = """
    This is the first try with the basic model.
"""

# PATH where we'll create the folder containig the new test.
TEST_FOLDER_PATH = "/homes/mcolombari/AI_for_Bioinformatics_Project/Training/Train_output"

# Load previous checkpoint.
START_FROM_CHECKPOINT = True
CHECKPOINT_PATH = "/homes/mcolombari/AI_for_Bioinformatics_Project/Training/Train_output/Train_Gene_25/model_checkpoints/Train_Gene_epoch_1.pth"

# Load data path
PATH_GTF_FILE = "/homes/mcolombari/AI_for_Bioinformatics_Project/Personal/gencode.v47.annotation.gtf"
PATH_FOLDER_GENE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression"
PATH_CASE_ID_STRUCTURE = "/homes/mcolombari/AI_for_Bioinformatics_Project/Preprocessing/Final/case_id_and_structure.json"


#   Model parameter TODO
hyperparameter = {
    'num_classes': 2,
    'epochs': 4,
    'batch_size': 10,
    'seed': 123456,
    'num_workers': 6,
    'lr': 0.01,
    'save_model_period': 2, # How many epoch to wait before save the next model.
    'percentage_of_test': 0.3, # How many percentage of the dataset is used for testing.
    'feature_to_save': ['tpm_unstranded'], # Specifci parameter for gene.
    'feature_to_compare': 'tpm_unstranded'
}

torch.manual_seed(hyperparameter['seed'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://pytorch-geometric.readthedocs.io/en/2.5.3/notes/create_dataset.html
lpd = LPD(PATH_GTF_FILE, PATH_FOLDER_GENE, PATH_CASE_ID_STRUCTURE,
          hyperparameter['feature_to_save'], hyperparameter['feature_to_compare'],
          hyperparameter['num_classes'], hyperparameter['percentage_of_test'])
data_train_list, data_test_list = lpd.get_data()  # List of Data.
# Inside of data we need to specify which y we have.

train_loader = DataLoader(data_train_list, batch_size=hyperparameter['batch_size'], shuffle=True, num_workers=hyperparameter['num_workers'], pin_memory=True)
test_loader = DataLoader(data_test_list, batch_size=hyperparameter['batch_size'], shuffle=True, num_workers=hyperparameter['num_workers'], pin_memory=True)
# pin_memory=True will automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.
# https://pytorch.org/docs/stable/data.html.


node_feature_number = 1
model = simple_GCN(node_feature_number, 10, hyperparameter['num_classes'])

s_epoch = 0
if START_FROM_CHECKPOINT:
    checkpoint = torch.load(CHECKPOINT_PATH)
    s_epoch = checkpoint['epoch']
    model_dict = checkpoint['model_dict']
    model.load_state_dict(model_dict)

# Create Folder and first files.
sm = SM(TEST_FOLDER_PATH, TEST_NAME)
sm.save_test_info(MORE_INFO, START_FROM_CHECKPOINT, CHECKPOINT_PATH)
sm.save_model_hyperparameter(hyperparameter)
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
sm.save_model_architecture(model)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameter['lr'])
criterion = torch.nn.CrossEntropyLoss()
# Here you could also use a scheduler to validate the model.


def train(loader):
    model.train()
    for data in loader:
        # Get the inputs and labels
        inputs, labels = data.x.unsqueeze(1).to(device), data.y.to(device)
        edge_index, batch = data.edge_index.to(device), data.batch.to(device)
        optimizer.zero_grad()

        # print(inputs.size())
        # print(labels.min(), labels.max())

        # Forward
        outputs = model(inputs, edge_index, batch)
        if isinstance(outputs, list):
            outputs = outputs[0] #check the model gets back only one output

        # Compute the loss
        loss = criterion(outputs, labels.squeeze())
        
        # Backward & optimize
        loss.backward()
        optimizer.step()
    

def test(loader):
    model.eval()
    losses = []
    all_label = []
    all_pred = []

    with torch.no_grad():
        for data in loader:
            # get the inputs and labels
            inputs, labels = data.x.unsqueeze(1).to(device), data.y.to(device)
            edge_index, batch = data.edge_index.to(device), data.batch.to(device)

            # forward
            outputs = model(inputs, edge_index, batch)

            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)

        # Compute the average loss & accuracy
        test_loss = sum(losses)/len(losses)
        all_label = torch.stack(all_label, dim=0)
        all_pred = torch.stack(all_pred, dim=0)
        test_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())

    return test_loss, test_acc



for epoch_index in range(s_epoch, hyperparameter['epochs']):
    print(f"Epoch {epoch_index + 1}")
    train(train_loader)
    train_loss, train_acc = test(train_loader)
    test_loss, test_acc = test(test_loader)
    print(f"\tTrain loss: {train_loss}")
    print(f"\tTrain acc: {train_acc}")
    print(f"\tTest loss: {test_loss}")
    print(f"\tTest acc: {test_acc}")
    sm.save_epoch_data(epoch_index, train_loss, train_acc, test_loss, test_acc)

    if (epoch_index - s_epoch) % hyperparameter['save_model_period'] == 0:
        print("###    Model saved    ###")
        sm.save_epoch(epoch_index, model)