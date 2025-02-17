import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import torch
import json
import os
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx, from_networkx

PATH_FOLDER_COPY_NUMBER = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/CopyNumber"
PATH_CASE_ID_STRUCTURE = "/homes/dlupo/Progetto_BioInformatics/AI_for_Bioinformatics_Project/Preprocessing/Final/case_id_and_structure.json"
PATH_GENE_ID_PROTEIN_CODING = "/homes/dlupo/Progetto_BioInformatics/AI_for_Bioinformatics_Project/Preprocessing/Final/gene_id_protein_coding.json"

with open(PATH_CASE_ID_STRUCTURE, 'r') as file:
    file_parsed = json.load(file)

copy_number_folder_list = []
os_list = []
for key in file_parsed.keys():
    copy_number_folder_list.append(file_parsed[key]["files"]["copy_number"])
    os_list.append(file_parsed[key]['os'])
print()
gene_id_list = []
with open(PATH_GENE_ID_PROTEIN_CODING) as json_file:
   gene_id_list = json.load(json_file)

def remove_version(x):
    if '.' in x:
        return x.split('.')[0]
    return x

list_df_CNV = []
for root, dirs, files in os.walk(PATH_FOLDER_COPY_NUMBER):
    for dir in dirs:
        for root, dirs, files in os.walk(PATH_FOLDER_COPY_NUMBER + "/" + dir):
            for file in files:
                if file in copy_number_folder_list:
                    parsed_file = pd.read_csv(PATH_FOLDER_COPY_NUMBER + "/" + dir + "/" + file, sep='\t')
                    parsed_file['gene_id'] = parsed_file['gene_id'].apply(remove_version)
                    parsed_file = parsed_file[parsed_file['gene_id'].isin(gene_id_list)]
                    list_df_CNV.append(parsed_file)

list_df_CNV_filled = []
for i in range(len(list_df_CNV)):
    list_df_CNV_filled.append(list_df_CNV[i].fillna(0))
        
os_list_sorted = list(os_list)
os_list_sorted.sort()
n = len(os_list_sorted)
num_classes = 2
split_values = []

for c in range(1, num_classes + 1):
    if c == num_classes:
        split_values.append(os_list_sorted[len(os_list_sorted) - 1])
    else:
        index = (n // num_classes) * c
        split_values.append(os_list_sorted[index - 1])


<<<<<<< HEAD
edge = [[],[]]
nodes = len(list_df_CNV_filled[0])
df = list_df_CNV_filled[0]
# Aggiunta delle connessioni in base alla sovrapposizione
for f_1_index in range(nodes):
    for f_2_index in range(f_1_index + 1, nodes):
        row1 = df.iloc[f_1_index]
        row2 = df.iloc[f_2_index]
        if row1['chromosome'] == row2['chromosome']:  # Evita duplicati
            if not (row1['end'] < row2['start'] or row2['end'] < row1['start']):
                edge[0].append(f_1_index)
                edge[0].append(f_2_index)
                edge[1].append(f_2_index)
                edge[1].append(f_1_index)

list_label = []
list_attribute = []
graph_indicator = []
list_of_list = []
count = 0
for case_index in range(226): 
    df = list_df_CNV_filled[case_index]

    temp_list = [[],[]]
    temp_list[0] = [elem + count for elem in edge[0]]
    temp_list[1] = [elem + count for elem in edge[1]]
    list_of_list.append(temp_list)
    count += nodes

=======
list_of_Data = []
list_label = []
list_attribute = []
edges = [[],[]]
count = 0
graph_indicator = []

for case_index in range(226): 
    df = list_df_CNV_filled[case_index]

>>>>>>> 731c396f8b463a753fcabd0637b7936a50b70355
    for i in range(len(df)):
        graph_indicator.append(case_index)

    if os_list[case_index] <= split_values[0]:
        list_label.append(0)
    else:
        list_label.append(1)

<<<<<<< HEAD
    for _, row in df.iterrows():
        list_attribute.append(row['copy_number'])

edges = [[],[]]
for item in list_of_list:
    edges[0].extend(item[0])
    edges[1].extend(item[1])

=======
    # Aggiunta delle connessioni in base alla sovrapposizione
    nodes = len(df)
    for f_1_index in range(nodes):
        for f_2_index in range(f_1_index + 1, nodes):
            row1 = df.iloc[f_1_index]
            row2 = df.iloc[f_2_index]
            if row1['chromosome'] == row2['chromosome']:  # Evita duplicati
                if not (row1['end'] < row2['start'] or row2['end'] < row1['start']):
                    edges[0].append(f_1_index + count)
                    edges[0].append(f_2_index + count)
                    edges[1].append(f_2_index + count)
                    edges[1].append(f_1_index + count)
                    # G.add_edge(row1['gene_name'], row2['gene_name'])

    count += nodes
    print(case_index)
    # list_attribute.append(row['copy_number'] for _, row in df_CNV.iterrows())
    for _, row in df.iterrows():
        list_attribute.append(row['copy_number'])

>>>>>>> 731c396f8b463a753fcabd0637b7936a50b70355
with open('./datasets/Copy_Number/Copy_Number_graph_labels.txt','w') as file:
    for item in list_label:
        file.write(str(item)+'\n')

with open('./datasets/Copy_Number/Copy_Number_node_attributes.txt','w') as file:
    for item in list_attribute:
        file.write(str(item)+'\n')

with open('./datasets/Copy_Number/Copy_Number_graph_indicator.txt','w') as file:
    for item in graph_indicator:
        file.write(str(item)+'\n')

with open('./datasets/Copy_Number/Copy_Number_A.txt','w') as file:
    for i in range(len(edges[0])):
        file.write(str(edges[0][i])+', '+str(edges[1][i])+'\n')


# Stampa gli edges
print("edges =", len(edges[0]))
