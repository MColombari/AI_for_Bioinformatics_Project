import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import torch
import json
import os
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx, from_networkx
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

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
                    attributes = ['chromosome','gene_name','start','end','min_copy_number','max_copy_number']
                    parsed_file = parsed_file.drop(columns=attributes)
                    parsed_file['gene_id'] = parsed_file['gene_id'].apply(remove_version)
                    parsed_file = parsed_file[parsed_file['gene_id'].isin(gene_id_list)].fillna(0)
                    list_df_CNV.append(parsed_file)

# Concatenare tutti i dataframe
df_concatenato = pd.concat(list_df_CNV)

# Calcolare la varianza per ogni gene_id
varianze = df_concatenato.groupby('gene_id')['copy_number'].var()

top_n = 1000  # numero di geni che si vuole mantenere
gene_significativi = varianze.nlargest(top_n).index 

scaler = StandardScaler()
list_df_CNV_filtered = []
for i in range(len(list_df_CNV)):
    df = list_df_CNV[i][list_df_CNV[i]['gene_id'].isin(gene_significativi)]
    X_scaled = scaler.fit_transform(df['copy_number'].values.reshape(-1,1))
    df = df.drop(columns='copy_number')
    df['copy_number'] = X_scaled.flatten()
    list_df_CNV_filtered.append(df)
     
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


THRESHOLD = 0.06
feature_to_compare = 'copy_number'
list_of_list = []
list_edges = []
count = 1
for case_index in range(0, len(list_df_CNV_filtered)):
    edges = []
    in_1 = [[v] for v in list(list_df_CNV_filtered[case_index][feature_to_compare])]

    dist_a = pairwise_distances(in_1, metric="euclidean")

    d_mask = np.zeros(dist_a.shape, dtype=bool)
    np.fill_diagonal(d_mask, 1)

    # Force the diagonal to be equal to Threshold, so it will not be considered, so no self loops.
    dist_a[d_mask] = THRESHOLD

    row, cols = np.where(dist_a < THRESHOLD)
    edges.append(list(row))
    edges.append(list(cols))
    list_edges.append(len(edges[0]))

    temp_list = [[],[]]
    temp_list[0] = [elem + count for elem in edges[0]]
    temp_list[1] = [elem + count for elem in edges[1]]
    list_of_list.append(temp_list)
    count += top_n


list_label = []
list_attribute = []
graph_indicator = []
for case_index in range(len(list_df_CNV_filtered)): 
    df = list_df_CNV_filtered[case_index]

    for i in range(len(df)):
        graph_indicator.append(case_index+1)

    if os_list[case_index] <= split_values[0]:
        list_label.append(0)
    else:
        list_label.append(1)

    for _, row in df.iterrows():
        list_attribute.append(row['copy_number'])

edges = [[],[]]
for item in list_of_list:
    edges[0].extend(item[0])
    edges[1].extend(item[1])

with open('./datasets/COPY_NUMBER/COPY_NUMBER_graph_labels.txt','w') as file:
    for item in list_label:
        file.write(str(item)+'\n')

with open('./datasets/COPY_NUMBER/COPY_NUMBER_node_attributes.txt','w') as file:
    for item in list_attribute:
        file.write(str(item)+'\n')

with open('./datasets/COPY_NUMBER/COPY_NUMBER_graph_indicator.txt','w') as file:
    for item in graph_indicator:
        file.write(str(item)+'\n')

with open('./datasets/COPY_NUMBER/COPY_NUMBER_A.txt','w') as file:
    for i in range(len(edges[0])):
        file.write(str(edges[0][i])+', '+str(edges[1][i])+'\n')

