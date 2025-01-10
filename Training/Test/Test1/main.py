import torch
from torch_geometric.data import Data
import json
from pathlib import Path
import os
import pandas as pd

def load_data(path_white_list, path_gene_expression_json, path_data):
    # Load case ID from white list
    white_list = []
    with open(path_white_list, 'r') as file:
        file_parsed = json.load(file)
        for case_id in file_parsed:
            white_list.append(case_id)

    # Load file name relative to case id in json file in gene expression.
    file_names = []
    with open(path_gene_expression_json, 'r') as file:
        file_parsed = json.load(file)
        for case in file_parsed:
            case_id = case["cases"][0]["case_id"]
            if case_id in white_list:
                file_names.append(case["file_name"])

    count = 0
    # Now explore data path to get the right files
    for root, dirs, files in os.walk(path_data):
        for dir in dirs:
            for root, dirs, files in os.walk(path_data + "/" + dir):
                for file in files:
                    if file in file_names:
                        # Parse file into graph
                        parsed_file = pd.read_csv(path_data + "/" + dir + "/" + file, sep='\t')
                        parsed_file['tpm_unstranded']

                        count = count + 1
                        print(file_names)
    



load_data("/homes/mcolombari/AI_for_Bioinformatics_Project/Preprocessing/Tests/Case_Id_Intersection/Test_3/white_list.json",
          "/homes/mcolombari/AI_for_Bioinformatics_Project/Preprocessing/Tests/Case_Id_Intersection/Test_3/GeneExpressionWithoutSolidTissue.json",
          "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression")