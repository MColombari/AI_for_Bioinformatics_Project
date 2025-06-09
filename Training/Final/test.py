from models import simple_GCN
import torch
from Load_and_Process_Data import LPD, LPDHybrid
from torch_geometric.loader import DataLoader
from collections import OrderedDict
from sklearn.metrics import accuracy_score
import numpy as np
from Save_model import SaveTest as ST

TEST_NAME = "Test_Gene"
TEST_FOLDER_PATH = "/homes/mcolombari/AI_for_Bioinformatics_Project/Training/Test_output"

CHECKPOINT_PATH = "/homes/mcolombari/AI_for_Bioinformatics_Project/Training/Train_output/Train_Gene_18/model_checkpoints/Train_Gene_epoch_99.pth"
PATH_GTF_FILE = "/homes/mcolombari/AI_for_Bioinformatics_Project/Personal/gencode.v47.annotation.gtf"
PATH_FOLDER_GENE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression"
PATH_CASE_ID_STRUCTURE = "/homes/mcolombari/AI_for_Bioinformatics_Project/Preprocessing/Final/case_id_and_structure.json"

PATH_EDGE_FILE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/9606.protein.links.v12.0.ENSG.txt"

hyperparameter = {
    'num_classes': 2,
    'epochs': 200,
    'batch_size': 10,
    'seed': 123456,
    'num_workers': 6,
    'lr': 0.01,
    'save_model_period': 10, # How many epoch to wait before save the next model.
    'percentage_of_test': 0.3, # How many percentage of the dataset is used for testing.
    'feature_to_save': ['tpm_unstranded'], # Specifci parameter for gene.
    'feature_to_compare': 'tpm_unstranded'
}

torch.manual_seed(hyperparameter['seed'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = simple_GCN(1, 100, hyperparameter['num_classes'])

# Load Model
checkpoint = torch.load(CHECKPOINT_PATH)
model_dict = checkpoint['model_dict']
model.load_state_dict(model_dict)


# https://pytorch-geometric.readthedocs.io/en/2.5.3/notes/create_dataset.html
lpd = LPDHybrid(PATH_GTF_FILE, PATH_FOLDER_GENE, PATH_CASE_ID_STRUCTURE,
                            hyperparameter['feature_to_save'], hyperparameter['feature_to_compare'],
                            hyperparameter['num_classes'], hyperparameter['percentage_of_test'],
                            PATH_EDGE_FILE)

lpd.read_gtf_file()
lpd.preprocessing()
lpd.create_graph()

data_list = lpd.list_of_Data

st = ST(TEST_FOLDER_PATH, TEST_NAME)

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

res = []
for d in data_list:
    c = lpd.get_instance_class(d)

    model.eval()

    with torch.no_grad():
        input, label = d.x.unsqueeze(1).to(device), torch.tensor(c).to(device)
        edge_index = d.edge_index.to(device)

        # print(input.size())
        # print(label.size())
        # print(edge_index.size())

        output = model(input, edge_index)
        loss = criterion(output[0], label)
        # collect labels & prediction
        prediction = torch.argmax(output[0])
        # print(output)
        # print(output[0])
        # print(prediction)

    res.append(f"OS: '{d.y}'\tclass: '{c}'\tloss: '{loss}'\tprediction: '{prediction}'")
st.save_results(res)