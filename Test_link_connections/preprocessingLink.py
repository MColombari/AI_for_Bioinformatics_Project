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

ggLink = pd.read_csv('ggLink.csv', sep='\t')
print(ggLink[['gene1','gene2','minResCount']])

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
for key in file_parsed.keys():
    copy_number_folder_list.append(file_parsed[key]["files"]["copy_number"])

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
                    parsed_file = parsed_file[parsed_file['gene_id'].isin(pc_set)]
                    list_df_CNV.append(parsed_file)

list_df_CNV_filled = []
for i in range(len(list_df_CNV)):
    list_df_CNV_filled.append(list_df_CNV[i].fillna(0))

#SOLUZIONE 1 : semplice, non ottimizzata
# edges = [[],[]]
# # nodes = len(list_df_CNV_filled[0])
# nodes = 50
# df = list_df_CNV_filled[0]
# start_time = time.time()  ########

# # Aggiunta delle connessioni in base alla sovrapposizione
# for f_1_index in range(nodes):
#     for f_2_index in range(f_1_index + 1, nodes):
#         row1 = df.iloc[f_1_index]
#         row2 = df.iloc[f_2_index]

#         df_gene = gtf_pc_no_dup[gtf_pc_no_dup['gene_id'] == row1['gene_id']]['gene_name']
#         if not df_gene.empty:
#             gene_name1 = df_gene.values[0]
#         else:
#             gene_name1 = ''

#         df_gene = gtf_pc_no_dup[gtf_pc_no_dup['gene_id'] == row2['gene_id']]['gene_name']
#         if not df_gene.empty:
#             gene_name2 = df_gene.values[0]
#         else:
#             gene_name2 = ''

#         row = ggLink[(ggLink['gene1'] == gene_name1) & (ggLink['gene2'] == gene_name2)]
#         if not row.empty:
#             if row['minResCount'].values[0] > 0:
#                 edges[0].append(f_1_index)
#                 edges[0].append(f_2_index)
#                 edges[1].append(f_2_index)
#                 edges[1].append(f_1_index)
            
# print(f" terminated in \t\t{np.floor(time.time() - start_time)}s") #time=257s
# print(edges)
# with open('/homes/dlupo/Progetto_BioInformatics/AI_for_Bioinformatics_Project/Test_link_connections/edges.txt','w') as file:
#     for i in range(len(edges[0])):
#         file.write(str(edges[0][i])+', '+str(edges[1][i])+'\n')


#SOLUZIONE 2 : un po ottimizzata
# # nodes = len(list_df_CNV_filled[0])
# nodes = 50
# df = list_df_CNV_filled[0]
# start_time = time.time()  ########
# edges = set()
# # Creazione di un dizionario per il gene_name
# gene_name_dict = dict(zip(gtf_pc_no_dup['gene_id'], gtf_pc_no_dup['gene_name']))

# # Preparazione delle colonne per il join
# df['gene_name'] = df['gene_id'].map(gene_name_dict)
# ggLink_filtered = ggLink[ggLink['minResCount'] > 0]

# # Aggiunta delle connessioni in base alla sovrapposizione
# for f_1_index in range(nodes):
#     print(f_1_index)
#     for f_2_index in range(f_1_index + 1, nodes):
#         gene_name1 = df.iloc[f_1_index]['gene_name']
#         gene_name2 = df.iloc[f_2_index]['gene_name']

#         if (ggLink_filtered[(ggLink_filtered['gene1'] == gene_name1) & (ggLink_filtered['gene2'] == gene_name2)].any(axis=None)):
#             edges.add((f_1_index, f_2_index))
#             edges.add((f_2_index, f_1_index))
            
# print(f" terminated in \t\t{np.floor(time.time() - start_time)}s")  #time=120s
# edges = [list(edge) for edge in zip(*edges)]
# print(edges)

#SOLUZIONE 3 : ottimizzata
nodes = len(list_df_CNV_filled[0])
# nodes = 50
edges = set()
df = list_df_CNV_filled[0]

start_time = time.time()  ########
# Creazione di un dizionario per il gene_name
gene_name_dict = dict(zip(gtf_pc_no_dup['gene_id'], gtf_pc_no_dup['gene_name']))

# Filtraggio iniziale per minResCount > 0
ggLink_filtered = ggLink[ggLink['minResCount'] > 0]

# Creazione di un dizionario di connessioni possibili
gene_links = {}
for _, row in ggLink_filtered.iterrows():
    gene_links.setdefault(row['gene1'], set()).add(row['gene2'])

# Unione per le connessioni
for f_1_index in range(nodes):
    gene_name1 = df.iloc[f_1_index]['gene_name']
    if gene_name1 in gene_links:
        for f_2_index in range(f_1_index + 1, nodes):
            gene_name2 = df.iloc[f_2_index]['gene_name']
            if gene_name2 in gene_links[gene_name1]:
                edges.add((f_1_index, f_2_index))
                edges.add((f_2_index, f_1_index))

# Conversione edges in liste
edges = [list(edge) for edge in zip(*edges)]
print(f" terminated in \t\t{np.floor(time.time() - start_time)}s") #time=62s
print(edges)
with open('/homes/dlupo/Progetto_BioInformatics/AI_for_Bioinformatics_Project/Test_link_connections/edges.txt','w') as file:
    for i in range(len(edges[0])):
        file.write(str(edges[0][i])+', '+str(edges[1][i])+'\n')

#time for 19000 nodes = 147 minutes