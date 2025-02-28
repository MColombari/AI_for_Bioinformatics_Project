import torch
from Save_model import SaveModel as SM
from models import simple_GCN, small_GCN, GAT
from Load_and_Process_Data import LPD, LPDEdgeKnowledgeBased, LPDHybrid
from torch_geometric.loader import DataLoader
from collections import OrderedDict
from sklearn.metrics import accuracy_score
import numpy as np
import torch_geometric.transforms as T
from torch.nn import DataParallel

# So we have a structure of folder where we have a main folder containig all the test for
# each subgroup (Methylation, Gene, Copy number), and in each of these folder we have
# a folder for each test, inside all the results and checkpoint are saved.

#   Data Parameter

# Name of the test, like methylation or gene... .
TEST_NAME = "Train_Gene"
MORE_INFO = """
    This is the first try with the basic model.\nwith new LPD.
"""

# PATH where we'll create the folder containig the new test.
TEST_FOLDER_PATH = "/homes/mcolombari/AI_for_Bioinformatics_Project/Training/Train_output"

# Load previous checkpoint.
START_FROM_CHECKPOINT = False
CHECKPOINT_PATH = ""

# Load data path
PATH_GTF_FILE = "/homes/mcolombari/AI_for_Bioinformatics_Project/Personal/gencode.v47.annotation.gtf"
PATH_FOLDER_GENE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression"
PATH_CASE_ID_STRUCTURE = "/homes/mcolombari/AI_for_Bioinformatics_Project/Preprocessing/Final/case_id_and_structure.json"
# For edge similarity file.
PATH_EDGE_FILE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/9606.protein.links.v12.0.ENSG.txt"
PATH_COMPLETE_EDGE_FILE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/edge_T650.json"
PATH_EDGE_ORDER_FILE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/edge_node_order.json"


#   Model parameter TODO
hyperparameter = {
    'num_classes': 2,
    'epochs': 10,
    'batch_size': 4,
    'seed': 123456,
    'num_workers': 2,
    'lr': 0.01,
    'save_model_period': 2, # How many epoch to wait before save the next model.
    'percentage_of_test': 0.1, # How many percentage of the dataset is used for testing.
    'feature_to_save': ['tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded'], # Specify parameter for gene.
    'feature_to_compare': 'tpm_unstranded'
}

torch.manual_seed(hyperparameter['seed'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Start code")
# print(f"\tFree memory usage:      {torch.cuda.mem_get_info()[0]}")
# print(f"\tTotal available memory: {torch.cuda.mem_get_info()[1]}")

# torch.cuda.empty_cache()


# https://pytorch-geometric.readthedocs.io/en/2.5.3/notes/create_dataset.html
lpd = LPDEdgeKnowledgeBased(PATH_GTF_FILE, PATH_FOLDER_GENE, PATH_CASE_ID_STRUCTURE,
                            hyperparameter['feature_to_save'], hyperparameter['feature_to_compare'],
                            hyperparameter['num_classes'], hyperparameter['percentage_of_test'],
                            PATH_EDGE_FILE, PATH_COMPLETE_EDGE_FILE, PATH_EDGE_ORDER_FILE)
data_train_list, data_test_list = lpd.get_data()  # List of Data.
# Inside of data we need to specify which y we have.

# Transform in sparse tensor.
# https://github.com/pyg-team/pytorch_geometric/issues/1702
# data_train_list = [T.ToSparseTensor()(data) for data in data_train_list]
# data_test_list = [T.ToSparseTensor()(data) for data in data_test_list]

train_loader = DataLoader(data_train_list, batch_size=hyperparameter['batch_size'], shuffle=True, num_workers=hyperparameter['num_workers'], pin_memory=True)
test_loader = DataLoader(data_test_list, batch_size=hyperparameter['batch_size'], shuffle=False, num_workers=hyperparameter['num_workers'], pin_memory=True)
# pin_memory=True will automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.
# https://pytorch.org/docs/stable/data.html.


node_feature_number = len(hyperparameter['feature_to_save'])
model = GAT(node_feature_number, 10, 10, hyperparameter['num_classes'], 0.2)
# model = DataParallel(model)

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
    index_batch = 0
    for data in loader:
        # print(f"Batch: {index_batch + 1}")
        index_batch += 1
        # Get the inputs and labels
        # https://github.com/pyg-team/pytorch_geometric/issues/1702
        data = T.ToSparseTensor()(data)
        if node_feature_number == 1:
            inputs, labels = data.x .unsqueeze(1).to(device), data.y.to(device)
        else:
            inputs, labels = data.x.to(device), data.y.to(device)
        edge_adj, batch = data.adj_t.to(device), data.batch.to(device)

        # print(f"Inputs:\t{inputs}")
        # print(f"Inputs size:\t{inputs.size()}")
        # print(f"Labels:\t{labels}")
        # print(f"Batch:\t{batch}")
        # print(f"Batch size:\t{batch.size()}")

        # print(inputs.size())
        # print(labels.min(), labels.max())

        # Forward
        outputs = model(inputs, edge_adj, batch)
        # if isinstance(outputs, list):
        #     outputs = outputs[0] #check the model gets back only one output

        # print(f"Output: {outputs}")
        # Compute the loss
        # print(f"Labels: {labels}")
        loss = criterion(outputs, labels)

        # print(f"Loss: {loss}")
        
        # Backward & optimize
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"{name} has no gradient!")
        optimizer.step()
        optimizer.zero_grad()
        # raise Exception("Stop")
    

def test(loader):
    model.eval()
    losses = []
    all_label = []
    all_pred = []

    with torch.no_grad():
        for data in loader:
            # get the inputs and labels
            data = T.ToSparseTensor()(data)
            if node_feature_number == 1:
                inputs, labels = data.x .unsqueeze(1).to(device), data.y.to(device)
            else:
                inputs, labels = data.x.to(device), data.y.to(device)
            edge_adj, batch = data.adj_t.to(device), data.batch.to(device)

            # print(data.x)
            # print(inputs)

            # forward
            outputs = model(inputs, edge_adj, batch)
            if isinstance(outputs, list):
                outputs = outputs[0] #check the model gets back only one output

            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.argmax(outputs, 1)
            # print(prediction)
            # print(prediction[0])
            # print(prediction[1])
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)

        # Compute the average loss & accuracy
        test_loss = sum(losses)/len(losses)
        all_label = torch.stack(all_label, dim=0)
        all_pred = torch.stack(all_pred, dim=0)
        test_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())

    return test_loss, test_acc



for epoch_index in range(s_epoch, hyperparameter['epochs']):
    print(f"\nEpoch {epoch_index + 1}")
    train(train_loader)
    print("\tAfter train")
    print(f"\t\tFree memory usage:      {torch.cuda.mem_get_info()[0]}")
    print(f"\t\tTotal available memory: {torch.cuda.mem_get_info()[1]}")
    train_loss, train_acc = test(train_loader)
    test_loss, test_acc = test(test_loader)
    print("\tAfter test")
    print(f"\t\tFree memory usage:      {torch.cuda.mem_get_info()[0]}")
    print(f"\t\tTotal available memory: {torch.cuda.mem_get_info()[1]}")
    print(f"\tTrain loss: {train_loss}")
    print(f"\tTrain acc: {train_acc}")
    print(f"\tTest loss: {test_loss}")
    print(f"\tTest acc: {test_acc}")
    sm.save_epoch_data(epoch_index, train_loss, train_acc, test_loss, test_acc)

    if (epoch_index + 1 - s_epoch) % hyperparameter['save_model_period'] == 0:
        print("###    Model saved    ###")
        sm.save_epoch(epoch_index + 1, model)