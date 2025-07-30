import torch
from Save_model import SaveModel as SM
from models import simple_GCN, bigger_GCN, small_GCN, GAT, SimpleGAT, ComplexGAT, EdgeAttrGNN, EdgeAttrGAT, EdgeAttrGNNLight
from Load_and_Process_Data import LPDEdgeKnowledgeBased
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
import yaml
import numpy as np
import sys

# So we have a structure of folder where we have a main folder containig all the test for
# each subgroup (Methylation, Gene, Copy number), and in each of these folder we have
# a folder for each test, inside all the results and checkpoint are saved.

#   Data Parameter loaded from YAML file.

with open("/homes/mcolombari/AI_for_Bioinformatics_Project/Training/Final_Regression/config.yaml", "r") as yamlfile:
    data_yaml = yaml.load(yamlfile, Loader=yaml.FullLoader)

# Name of the test, like methylation or gene... .
TEST_NAME = data_yaml['Generic']['Test_name']
MORE_INFO = data_yaml['Generic']['More_info']

# PATH where we'll create the folder containig the new test.
TEST_FOLDER_PATH = data_yaml['PATH']['Test_folder_path']

# Load previous checkpoint.
START_FROM_CHECKPOINT = data_yaml['Conditions']['Start_from_checkpoint']
CHECKPOINT_PATH = data_yaml['PATH']['Checkpoint_path']

# Load Dataset.
LOAD_DATASET = data_yaml['Conditions']['Load_dataset']
DATASET_FROM_FOLDER_PATH = data_yaml['PATH']['Dataset_from_folder_path']
SAVE_DATASET = data_yaml['Conditions']['Save_dataset']

# Train or not
DONT_TRAIN = data_yaml['Conditions']['Dont_train']

# Load data path
PATH_GTF_FILE = data_yaml['Generic_PATH']['pathGtfFile']
PATH_FOLDER_GENE = data_yaml['Generic_PATH']['pathFolderGene']
PATH_FOLDER_COPY_NUMBER = data_yaml['Generic_PATH']['pathFolderCopyNumber']
PATH_FOLDER_METHYLATION = data_yaml['Generic_PATH']['pathFolderMethylation']
PATH_CASE_ID_STRUCTURE = data_yaml['Generic_PATH']['pathCaseIdStructure']

PATH_METHYLATION_CONVERTER = data_yaml['Generic_PATH']['pathMethylationConverter']

# For edge similarity files.
PATH_EDGE_FILE = data_yaml['Generic_PATH']['pathEdgeFile']

# Order of nodes files.
PATH_ORDER_GENE = data_yaml['Generic_PATH']['pathOrderGene']

# Test and Train separation file.
PATH_TEST_CLASS = data_yaml['Generic_PATH']['pathTestClass']
PATH_TRAIN_CLASS = data_yaml['Generic_PATH']['pathTrainClass']

#   Model parameter TODO
hyperparameter = data_yaml['hyperparameter']

torch.manual_seed(hyperparameter['seed'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Start code")
# print(f"\tFree memory usage:      {torch.cuda.mem_get_info()[0]}")
# print(f"\tTotal available memory: {torch.cuda.mem_get_info()[1]}")

# torch.cuda.empty_cache()

sm = SM(TEST_FOLDER_PATH, TEST_NAME)

if LOAD_DATASET:
    train_loader = torch.load(f"{DATASET_FROM_FOLDER_PATH}/datasets/train.pkl", weights_only=False)
    test_loader = torch.load(f"{DATASET_FROM_FOLDER_PATH}/datasets/test.pkl", weights_only=False)

    # first_batch = next(iter(train_loader))

    # # If it's a batch of graphs, take the first one only
    # if hasattr(first_batch, 'num_graphs') and first_batch.num_graphs > 1:
    #     data = first_batch.to_data_list()[0]
    # else:
    #     data = first_batch

    # # Repeat it `repeat` times
    # repeated_data = [data.clone() for _ in range(100)]

    # # Create new DataLoader with just this repeated graph
    # # DataLoader(repeated_data, batch_size=batch_size, shuffle=True)
    # train_loader = DataLoader(repeated_data, batch_size=hyperparameter['batch_size'], shuffle=True, num_workers=hyperparameter['num_workers'], pin_memory=True)

else:
    # https://pytorch-geometric.readthedocs.io/en/2.5.3/notes/create_dataset.html
    lpd = LPDEdgeKnowledgeBased(PATH_GTF_FILE, PATH_FOLDER_GENE, PATH_FOLDER_METHYLATION,
                                PATH_FOLDER_COPY_NUMBER, PATH_CASE_ID_STRUCTURE, PATH_METHYLATION_CONVERTER,
                                PATH_TEST_CLASS, PATH_TRAIN_CLASS, PATH_EDGE_FILE, 
                                PATH_ORDER_GENE, 
                                hyperparameter['feature_to_save'], hyperparameter['num_nodes'],
                                sm)
    data_train_list, data_test_list = lpd.get_data()  # List of Data.
    # Inside of data we need to specify which y we have.

    # Transform in sparse tensor.
    # https://github.com/pyg-team/pytorch_geometric/issues/1702
    # data_train_list = [T.ToSparseTensor()(data) for data in data_train_list]
    # data_test_list = [T.ToSparseTensor()(data) for data in data_test_list]

    # Why pin memory set to true? Answer -> https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    train_loader = DataLoader(data_train_list, batch_size=hyperparameter['batch_size'], shuffle=True, num_workers=hyperparameter['num_workers'], pin_memory=True)
    test_loader = DataLoader(data_test_list, batch_size=hyperparameter['batch_size'], shuffle=True, num_workers=hyperparameter['num_workers'], pin_memory=True)

    if SAVE_DATASET:
        sm.save_dataset(train_loader, test_loader)
# pin_memory=True will automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.
# https://pytorch.org/docs/stable/data.html.

# This flag is used to just create the dataset and dont train the model.
if DONT_TRAIN:
    sys.exit(0)

node_feature_number = 0
for k in hyperparameter['feature_to_save'].keys():
    node_feature_number += len(hyperparameter['feature_to_save'][k])

sm.print(f"\nNumber of input feature: {node_feature_number}")
    
# model = simple_GCN(node_feature_number, hyperparameter['num_classes'])
# model = bigger_GCN(node_feature_number, hyperparameter['num_classes'])
# model = small_GCN(node_feature_number, 750, hyperparameter['num_classes'])
model = EdgeAttrGNN(node_feature_number, 64, hyperparameter['num_classes'])
# model = EdgeAttrGNNLight(node_feature_number, 128, hyperparameter['num_classes'])
# model = EdgeAttrGAT(node_feature_number, 15, hyperparameter['num_classes'], num_layers=1, heads=7)
# model = GAT(node_feature_number, 1000, 30, hyperparameter['num_classes'], 0.2)
# model = SimpleGAT(node_feature_number, 2000, 30, hyperparameter['num_classes'], 0.2)
# model = ComplexGAT(node_feature_number, 500, 20, hyperparameter['num_classes'], 0.2)

# model = DataParallel(model)

s_epoch = 0
if START_FROM_CHECKPOINT:
    checkpoint = torch.load(CHECKPOINT_PATH)
    s_epoch = checkpoint['epoch']
    model_dict = checkpoint['model_dict']
    model.load_state_dict(model_dict)

# Create Folder and first files.
sm.save_test_info(data_yaml)
sm.save_model_hyperparameter(hyperparameter)
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
sm.save_model_architecture(model)

model = model.to(device)
# model.half()

for name, param in model.named_parameters():
    if torch.isnan(param).any():
        raise Exception(f"NaN detected in {name} weights!")

optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameter['lr'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',         # we want to minimize validation loss
    factor=0.5,         # reduce LR by half
    patience=5,         # wait 5 epochs before reducing
)
loss_mse = torch.nn.MSELoss()
# Here you could also use a scheduler to validate the model.

def range_accuracy(labels: list, predictions: list, margin:int=120)->float:
    right = 0
    total = 0
    assert len(labels) == len(predictions)
    assert len(labels) > 0
    for i in range(len(labels)):
        l = labels[i]
        p = predictions[i]
        if (l - margin/2) <= p <= (l + margin/2):
            right += 1
        total += 1
    return right / total

def check_c_index_metrix(labels: list, predictions: list)->bool:
    l_indexs = np.argsort(labels)
    p_indexs = np.argsort(predictions)
    return np.array_equal(l_indexs, p_indexs)

# def cindex_loss(pred, y):
#     n = 0
#     loss = 0.0
#     if y.dim() == 0:
#         dim = 1
#     else:
#         dim = pred.shape[0]
#     # print(pred)
#     for i in range(dim):
#         for j in range(dim):
#             if y[i] < y[j]:
#                 diff = pred[j] - pred[i]
#                 loss += torch.sigmoid(diff)
#                 n += 1
#     return loss if n > 0 else torch.tensor(0.0, device=device)

def cindex_loss_vectorized(pred, y):
    pred = pred.view(-1)
    y = y.view(-1)

    # Create all pairwise differences
    # https://stackoverflow.com/questions/69797614/indexing-a-tensor-with-none-in-pytorch
    diff = pred[:, None] - pred[None, :]
    y_diff = y[:, None] - y[None, :]

    # Mask for valid pairs where y_i > y_j
    valid = y_diff > 0

    # Apply sigmoid to the negative prediction difference
    loss = torch.sigmoid(-diff[valid])

    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return loss.mean()

def train(loader):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    index_batch = 0
    for data in loader:
        # print(f"\tBatch: {index_batch + 1}")
        index_batch += 1
        # Get the inputs and labels
        # https://github.com/pyg-team/pytorch_geometric/issues/1702
        # data = T.ToSparseTensor()(data)
        if torch.isnan(data.x).any() or torch.isnan(data.y).any():
            raise Exception("NaN detected in inputs!")
        
        inputs, labels = data.x.to(device), data.y.to(device)
        # edge_adj, batch = data.adj_t.to(device), data.batch.to(device)
        edge_index, batch = data.edge_index.to(device), data.batch.to(device)
        edge_attr = data.edge_attr.to(device)

        if torch.isnan(edge_index).any() or torch.isinf(edge_index).any():
            raise Exception("NaN or Inf detected in edge_index!")
        num_nodes = inputs.shape[0]
        if (edge_index >= num_nodes).any() or (edge_index < 0).any():
            raise Exception("Invalid edge_index detected!")

        # print(f"Inputs:\t{inputs}")
        # print(f"Inputs size:\t{inputs.size()}")
        # print(f"Labels:\t{labels}")
        # print(f"Batch:\t{batch}")
        # print(f"Batch size:\t{batch.size()}")

        optimizer.zero_grad()

        # Forward
        # outputs = model(inputs, edge_index, batch)
        outputs = model(inputs, edge_index, edge_attr, batch)
        # if isinstance(outputs, list):
        #     outputs = outputs[0] #check the model gets back only one output
        # print(f"Output: {outputs}")
        if torch.isnan(outputs).any():
            raise Exception("NaN detected in model output!")

        # Compute the loss
        # print(f"Labels: {labels}")
        loss = cindex_loss_vectorized(outputs, labels) * hyperparameter['alpha_loss'] + loss_mse(outputs, labels)

        # print(f"Loss: {loss}")
        
        # Backward & optimize
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                pass
                # raise Exception(f"{name} has no gradient!")
        for name, param in model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                raise Exception(f"NaN or Inf detected in gradients: {name}")
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        # raise Exception("Stop")
    

def test(loader):
    model.eval()
    losses = []
    all_label = []
    all_pred = []
    c_index_result = []

    with torch.no_grad():
        for data in loader:
            # get the inputs and labels
            # data = T.ToSparseTensor()(data)
            inputs, labels = data.x.to(device), data.y.to(device)
            # edge_adj, batch = data.adj_t.to(device), data.batch.to(device)
            edge_index, batch = data.edge_index.to(device), data.batch.to(device)
            edge_attr = data.edge_attr.to(device)

            # print(f"Inputs:\t{inputs}")
            # print(f"Inputs size:\t{inputs.size()}")
            # print(f"Labels:\t{labels}")
            # print(f"Batch:\t{batch}")
            # print(f"Batch size:\t{batch.size()}")

            # forward
            outputs = model(inputs, edge_index, edge_attr, batch)
            # if isinstance(outputs, list):
            #     outputs = outputs[0] #check the model gets back only one output

            # print(f"Output: {outputs}")

            # compute the loss
            loss = cindex_loss_vectorized(outputs, labels) * hyperparameter['alpha_loss'] + loss_mse(outputs, labels)
            losses.append(loss.item())
            
            # print(f"Loss: {loss}")

            # collect labels & prediction
            prediction = outputs
            # print(prediction)
            # print(prediction[0])
            # print(prediction[1])
            all_label.extend(labels)
            all_pred.extend(prediction)

            c_index_result.append(1 if check_c_index_metrix(labels.cpu().tolist(), prediction.squeeze().cpu().tolist()) else 0)

        # Compute the average loss & accuracy
        test_loss = sum(losses)/len(losses)
        all_label = torch.stack(all_label, dim=0)
        all_pred = torch.stack(all_pred, dim=0)
        # test_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
        test_acc = range_accuracy(all_label.cpu().tolist(), all_pred.squeeze().cpu().tolist())
        test_c_index = sum(c_index_result) / len(c_index_result)

    return test_loss, test_acc, test_c_index



for epoch_index in range(s_epoch, hyperparameter['epochs']):
    sm.print(f"\nEpoch {epoch_index + 1}")
    train(train_loader)
    sm.print("\tAfter train")
    # sm.print(f"\t\tFree memory usage:      {torch.cuda.mem_get_info()[0]}")
    # sm.print(f"\t\tTotal available memory: {torch.cuda.mem_get_info()[1]}")
    train_loss, train_acc, train_c_index = test(train_loader)
    test_loss, test_acc, test_c_index = test(test_loader)

    scheduler.step(test_loss)  # This will update LR.

    sm.print("\tAfter test")
    sm.print(f"\t\tFree memory usage:      {torch.cuda.mem_get_info()[0]}")
    sm.print(f"\t\tTotal available memory: {torch.cuda.mem_get_info()[1]}")
    sm.print(f"\tTrain loss: {train_loss}")
    sm.print(f"\tTrain acc: {train_acc}")
    sm.print(f"\tTrain c index: {train_c_index}")
    sm.print(f"\tTest loss: {test_loss}")
    sm.print(f"\tTest acc: {test_acc}")
    sm.print(f"\tTest c index: {test_c_index}")
    sm.save_epoch_data(epoch_index, train_loss, train_acc, train_c_index, test_loss, test_acc, test_c_index)

    if (epoch_index + 1 - s_epoch) % hyperparameter['save_model_period'] == 0:
        sm.print("###    Model saved    ###")
        sm.save_epoch(epoch_index + 1, model)
