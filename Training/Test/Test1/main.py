import torch
from torch_geometric.data import Data
import json
from pathlib import Path
import os
import pandas as pd

def load_data(path_white_list, path_gene_expression_json, path_data, path_relation):
    # Load case ID from white list
    white_list = []
    with open(path_white_list, 'r') as file:
        file_parsed = json.load(file)
        for case_id in file_parsed:
            white_list.append(case_id)

    # Load file name relative to case id in json file in gene expression.
    dict_file_name_to_case_id_relation = {}
    file_names = []
    with open(path_gene_expression_json, 'r') as file:
        file_parsed = json.load(file)
        for case in file_parsed:
            case_id = case["cases"][0]["case_id"]
            if case_id in white_list:
                dict_file_name_to_case_id_relation[case['file_name']] = case_id
                file_names.append(case["file_name"])


    with open(path_relation, 'r') as file:
        dict_relation = json.load(file)

    count = 0
    list_list_featurevectors = []
    list_value = []
    # Now explore data path to get the right files
    for root, dirs, files in os.walk(path_data):
        for dir in dirs:
            for root, dirs, files in os.walk(path_data + "/" + dir):
                for file in files:
                    if file in file_names:
                        # Parse file into graph
                        parsed_file = pd.read_csv(
                            path_data + "/" + dir + "/" + file, sep='\t', comment="#", skiprows=lambda x: x in [2, 3, 4, 5])
                        
                        case_id = dict_file_name_to_case_id_relation[file]
                        if dict_relation[case_id]['type'] == 'dead':
                            list_value.append(int(dict_relation[dict_file_name_to_case_id_relation[file]]['value']))
                            list_list_featurevectors.append(list(parsed_file['tpm_unstranded']))
                            count = count + 1
                            print(count)
                        # print(file_names)

    x = list(zip(range(len(list_value)), list_value))

    x = sorted(x, key=lambda x: x[1])
    combinations = [[],[]]
    print(x)

    for i in range(len(x)):
        for j in range(i+1, len(x)):
            if(x[j][1] < x[i][1] + 3):
                combinations[0].append(x[i][0])
                combinations[0].append(x[j][0])
                combinations[1].append(x[j][0])
                combinations[1].append(x[i][0])
            else:
                break
                        
    edge_index = torch.tensor(combinations, dtype=torch.long)
    x = torch.tensor([list(a) for a in x], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    



load_data("/homes/mcolombari/AI_for_Bioinformatics_Project/Preprocessing/Tests/Case_Id_Intersection/Test_3/white_list.json",
          "/homes/mcolombari/AI_for_Bioinformatics_Project/Preprocessing/Tests/Case_Id_Intersection/Test_3/GeneExpressionWithoutSolidTissue.json",
          "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression",
          "/homes/mcolombari/AI_for_Bioinformatics_Project/Preprocessing/Final/case_id_to_value_relation.json")