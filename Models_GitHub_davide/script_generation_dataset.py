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
PATH_CASE_ID_STRUCTURE = "./case_id_and_structure.json"
PATH_GENE_ID_PROTEIN_CODING = "./AI_for_Bioinformatics_Project/Preprocessing/Final/gene_id_protein_coding.json"

with open(PATH_CASE_ID_STRUCTURE, 'r') as file:
    file_parsed = json.load(file)

copy_number_folder_list = []
os_list = []
for key in file_parsed.keys():
    copy_number_folder_list.append(file_parsed[key]["files"]["copy_number"])
    os_list.append(file_parsed[key]['os'])

number_of_rows = []
list_df_CNV = []
for root, dirs, files in os.walk(PATH_FOLDER_COPY_NUMBER):
    for dir in dirs:
        for root, dirs, files in os.walk(PATH_FOLDER_COPY_NUMBER + "/" + dir):
            for file in files:
                if file in copy_number_folder_list:
                    list_df_CNV.append(pd.read_csv(PATH_FOLDER_COPY_NUMBER + "/" + dir + "/" + file, sep='\t'))

gene_id_list = []
with open(PATH_GENE_ID_PROTEIN_CODING) as json_file:
   gene_id_list = json.load(json_file)

# Converti gene_id_list in un set per velocizzare le operazioni di lookup
gene_id_set = set(gene_id_list)
# Itera su ogni dataframe nella lista e verifica se i valori di 'gene_id' sono nel set
list_df_CNV_filtered = []
for df in list_df_CNV:
    # Filtra le righe del dataframe dove 'gene_id' è presente in gene_id_set
    df_filtered = df[df['gene_id'].isin(gene_id_set)]
    list_df_CNV_filtered.append(df_filtered)

list_df_CNV_filled = []
for i in range(len(list_df_CNV_filtered)):
    list_df_CNV_filled.append(list_df_CNV_filtered[i].fillna(0))

        
os_list.sort()
n = len(os_list)
num_classes = 2
split_values = []

for c in range(1, num_classes + 1):
    if c == num_classes:
        split_values.append(os_list[len(os_list) - 1])
    else:
        index = (n // num_classes) * c
        split_values.append(os_list[index - 1])

"""
Crea grafi per ogni caso nel dataset.
"""
list_of_Data = []
list_label = []
list_attribute = []
edges = [[],[]]
count = 0
for case_index in range(20): 
    df_CNV = list_df_CNV_filled[case_index][:100]

    # for i in range(len(df_CNV)):
    #     with open('./dataset/Copy_Number_graph_indicator.txt','a') as file:
    #         file.write(str(case_index)+'\n')

    if os_list[case_index] <= split_values[0]:
        list_label.append(0)
    else:
        list_label.append(1)

    # Crea un grafo vuoto
    G = nx.Graph()
    
    # Aggiungi i nodi con i loro attributi
    nodes_data = {row['gene_name']: {'x': row['copy_number']} 
                 for _, row in df_CNV.iterrows()}
    G.add_nodes_from(nodes_data.items())

    # Aggiunta delle connessioni in base alla sovrapposizione
    nodes = len(df_CNV)
    for f_1_index in range(count, nodes + count):
        for f_2_index in range(f_1_index + 1, nodes + count):
            row1 = df.iloc[f_1_index]
            row2 = df.iloc[f_2_index]
            if row1['chromosome'] == row2['chromosome']:  # Evita duplicati
                if not (row1['end'] < row2['start'] or row2['end'] < row1['start']):
                    edges[0].append(f_1_index)
                    edges[0].append(f_2_index)
                    edges[1].append(f_2_index)
                    edges[1].append(f_1_index)
                    # G.add_edge(row1['gene_name'], row2['gene_name'])

    count += nodes

    # list_attribute.append(row['copy_number'] for _, row in df_CNV.iterrows())
    for _, row in df_CNV.iterrows():
        list_attribute.append(row['copy_number'])

if not os.path.exists('./dataset/Copy_Number_graph_labels.txt'):
    with open('./dataset/Copy_Number_graph_labels.txt','w') as file:
        for item in list_label:
            file.write(str(item)+'\n')

if not os.path.exists('./dataset/Copy_Number_node_attributes.txt'):
    with open('./dataset/Copy_Number_node_attributes.txt','w') as file:
        for item in list_attribute:
            file.write(str(item)+'\n')

print(split_values)
# Stampa gli edges
print("edges =", len(edges[0]))
