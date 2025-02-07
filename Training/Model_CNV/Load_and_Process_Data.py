import pandas as pd
from pathlib import Path
import torch
import json
import os
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt

PATH_FOLDER_COPY_NUMBER = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/CopyNumber"
PATH_CASE_ID_STRUCTURE = "./case_id_and_structure.json"
PATH_GENE_ID_PROTEIN_CODING = "./gene_id_protein_coding.json"

class LPD:
    def __init__(self):
        pass

    # Here i have to split the dataset in train and test, while keeping balance
    # between all the label in each subset.
    # Return train and test separately.
    def get_data(self):
        with open(PATH_CASE_ID_STRUCTURE, 'r') as file:
            file_parsed = json.load(file)

        copy_number_folder_list = []
        os_list = []
        for key in file_parsed.keys():
            copy_number_folder_list.append(file_parsed[key]["files"]["copy_number"])
            os_list.append(file_parsed[key]['os'])

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

        list_df_CNV_filtered = []
        for df in list_df_CNV:
            # Filtra le righe del dataframe dove 'gene_id' è presente in gene_id_set
            df_filtered = df[df['gene_id'].isin(gene_id_set)]
            list_df_CNV_filtered.append(df_filtered)

        list_df_CNV_filled = []
        for i in range(len(list_df_CNV_filtered)):
            list_df_CNV_filled.append(list_df_CNV_filtered[i].fillna(0))

        list_of_Data = []
        for case_index in range(0,20):

            df_CNV = list_df_CNV_filled[case_index][:200]

            # Crea un grafo vuoto
            G = nx.Graph()

            # Aggiungi i geni come nodi
            for _, row in df_CNV.iterrows():
                G.add_node(row['gene_name'], x=row['copy_number'])

            # Aggiungi archi basati sulla sovrapposizione delle coordinate (start, end)
            for i, gene1 in df_CNV.iterrows():
                for j, gene2 in df_CNV.iterrows():
                    if i >= j:
                        continue  # Evita di considerare due volte la stessa coppia
                    if gene1['chromosome'] == gene2['chromosome']:
                        # Controlla la sovrapposizione dei segmenti
                        if (gene1['start'] <= gene2['end']) and (gene1['end'] >= gene2['start']):
                            G.add_edge(gene1['gene_name'], gene2['gene_name'])

            pyg_graph = from_networkx(G)

            if os_list[case_index] > 1000:
                pyg_graph['y'] = torch.tensor([1])
            else:
                pyg_graph['y'] = torch.tensor([0])

            list_of_Data.append(pyg_graph)
            # print(case_index)

        train_list = list_of_Data[0:15]
        test_list = list_of_Data[15:]

        return train_list, test_list