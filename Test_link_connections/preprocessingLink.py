import pandas as pd
import re
import networkx as nx
import numpy as np
from pathlib import Path
import torch
import json
import os
import time
import matplotlib.pyplot as plt
from torch_geometric.data import Data

ggLink = pd.read_csv('ggLink.csv', sep='\t')
# print(ggLink[['gene1','gene2','minResCount']])

def remove_version(x):
    if '.' in x:
        return x.split('.')[0]
    return x

gtf_file_path = '/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/gencode.v47.annotation.gtf'
gtf = pd.read_csv(gtf_file_path, sep="\t", header=None, comment='#')
gtf.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

parameters = ['gene_id', 'gene_type', 'gene_name']
for p in parameters:
    gtf[p] = gtf['attribute'].apply(lambda x: re.findall(rf'{p} "([^"]*)"', x)[0] if rf'{p} "' in x else np.nan)

gtf.drop(['attribute','source','score','strand','frame'], axis=1, inplace=True)

gtf['gene_id'] = gtf['gene_id'].apply(remove_version)

gtf_pc = gtf[gtf['gene_type'] == 'protein_coding']

# Protein coding set
pc_set = set(gtf_pc['gene_id'].to_list())

gtf_pc_no_dup = gtf_pc.drop_duplicates(subset=['gene_id'])

PATH_FOLDER_COPY_NUMBER = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/CopyNumber"
PATH_CASE_ID_STRUCTURE = "/homes/dlupo/Progetto_BioInformatics/AI_for_Bioinformatics_Project/Preprocessing/Final/case_id_and_structure.json"

with open(PATH_CASE_ID_STRUCTURE, 'r') as file:
    file_parsed = json.load(file)

copy_number_folder_list = []
os_list = []
for key in file_parsed.keys():
    copy_number_folder_list.append(file_parsed[key]["files"]["copy_number"])
    os_list.append(file_parsed[key]['os'])

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
                    parsed_file = parsed_file[parsed_file['gene_id'].isin(pc_set)].fillna(0)
                    list_df_CNV.append(parsed_file)

# Concatenare tutti i dataframe
df_concatenato = pd.concat(list_df_CNV)
# Calcolare la varianza per ogni gene_id
varianze = df_concatenato.groupby('gene_id')['copy_number'].var()

top_n = 1000  # numero di geni che si vuole mantenere
gene_significativi = varianze.nlargest(top_n).index

list_df_CNV_filtered = []
for i in range(len(list_df_CNV)):
    list_df_CNV_filtered.append(list_df_CNV[i][list_df_CNV[i]['gene_id'].isin(gene_significativi)])


start_time = time.time()  ########

#SOLUZIONE 2 : un po ottimizzata
nodes = len(list_df_CNV_filtered[0])
edges = set()
df = list_df_CNV_filtered[0]

# Creazione di un dizionario per il gene_name
gene_name_dict = dict(zip(gtf_pc_no_dup['gene_id'], gtf_pc_no_dup['gene_name']))
ggLink_filtered = ggLink[ggLink['minResCount'] > 50]

# Preparazione delle colonne per il join
df['gene_name'] = df['gene_id'].map(gene_name_dict)

# Aggiunta delle connessioni in base alla sovrapposizione
for f_1_index in range(nodes):
    for f_2_index in range(f_1_index + 1, nodes):
        gene_name1 = df.iloc[f_1_index]['gene_name']
        gene_name2 = df.iloc[f_2_index]['gene_name']

        if (ggLink_filtered[(ggLink_filtered['gene1'] == gene_name1) & (ggLink_filtered['gene2'] == gene_name2)].any(axis=None)):
            edges.add((f_1_index, f_2_index))
            edges.add((f_2_index, f_1_index))

edges = [list(edge) for edge in zip(*edges)]
with open('/homes/dlupo/Progetto_BioInformatics/AI_for_Bioinformatics_Project/Test_link_connections/edges2.txt','w') as file:
    for i in range(len(edges[0])):
        file.write(str(edges[0][i])+', '+str(edges[1][i])+'\n')

print(f" terminated in \t\t{np.floor(time.time() - start_time)}s") #time=34222s -> 9.5h
print(len(edges[0]))

# #SOLUZIONE 3 : ottimizzata
# nodes = len(list_df_CNV[0])
# # nodes = 50
# edges = set()
# df = list_df_CNV[0]

# start_time = time.time()  ########

# # Creazione di un dizionario per il gene_name
# gene_name_dict = dict(zip(gtf_pc_no_dup['gene_id'], gtf_pc_no_dup['gene_name']))

# # Filtraggio iniziale per minResCount > 0
# ggLink_filtered = ggLink[ggLink['minResCount'] > 0]

# # Creazione di un dizionario di connessioni possibili
# gene_links = {}
# for _, row in ggLink_filtered.iterrows():
#     gene_links.setdefault(row['gene1'], set()).add(row['gene2'])

# # Unione per le connessioni
# for f_1_index in range(nodes):
#     gene_name1 = df.iloc[f_1_index]['gene_name']
#     if gene_name1 in gene_links:
#         for f_2_index in range(f_1_index + 1, nodes):
#             gene_name2 = df.iloc[f_2_index]['gene_name']
#             if gene_name2 in gene_links[gene_name1]:
#                 edges.add((f_1_index, f_2_index))
#                 edges.add((f_2_index, f_1_index))

# # Conversione edges in liste
# edges = [list(edge) for edge in zip(*edges)]
# print(f" terminated in \t\t{np.floor(time.time() - start_time)}s") #time=62s
# print(edges)
# with open('/homes/dlupo/Progetto_BioInformatics/AI_for_Bioinformatics_Project/Test_link_connections/edges.txt','w') as file:
#     for i in range(len(edges[0])):
#         file.write(str(edges[0][i])+', '+str(edges[1][i])+'\n')

# #time for 19000 nodes = 147 minutes