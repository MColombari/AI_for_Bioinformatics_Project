import pandas as pd
from pathlib import Path
import torch
import json
import os
import time
import numpy as np
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt

def measure_time(func):
    def wrapper(x):
        start_time = time.time()
        func(x)
        print(f"\t\t{np.floor(time.time() - start_time)}s")
    return wrapper

class LPD:
    def __init__(self, folder_copy_number_path: str, case_id_json_path: str,
        gene_id_protein_coding_path: str, num_classes: int, percentage_test: float):
        self.folder_copy_number_path = folder_copy_number_path
        self.case_id_json_path = case_id_json_path\
        self.gene_id_protein_coding_path = gene_id_protein_coding_path

        self.THRESHOLD = 0.1
        self.num_classes = num_classes
        self.percentage_test = percentage_test

    # Here i have to split the dataset in train and test, while keeping balance
    # between all the label in each subset.
    # Return train and test separately.
    @measure_time
    def preprocessing(self):
        with open(self.case_id_json_path, 'r') as file:
            file_parsed = json.load(file)

        copy_number_folder_list = []
        self.os_list = []
        for key in file_parsed.keys():
            copy_number_folder_list.append(file_parsed[key]["files"]["copy_number"])
            self.os_list.append(file_parsed[key]['os'])

        list_df_CNV = []
        for root, dirs, files in os.walk(self.folder_copy_number_path):
            for dir in dirs:
                for root, dirs, files in os.walk(self.folder_copy_number_path + "/" + dir):
                    for file in files:
                        if file in copy_number_folder_list:
                            list_df_CNV.append(pd.read_csv(self.folder_copy_number_path + "/" + dir + "/" + file, sep='\t'))

        gene_id_list = []
        with open(self.gene_id_protein_coding_path) as json_file:
            gene_id_list = json.load(json_file)

        # Convert gene_id_list into a set data structure to speed up the lookup
        gene_id_set = set(gene_id_list)

        list_df_CNV_filtered = []
        for df in list_df_CNV:
            # Filter dataframe rows where 'gene_id'value is in gene_id_set
            df_filtered = df[df['gene_id'].isin(gene_id_set)]
            list_df_CNV_filtered.append(df_filtered)

        self.list_df_CNV_filled = []
        for i in range(len(list_df_CNV_filtered)):
            self.list_df_CNV_filled.append(list_df_CNV_filtered[i].fillna(0))

    def find_overlapping_genes(self, df_CNV):
        """
        Trova in modo efficiente le coppie di geni che si sovrappongono.
        """
        # Ordina il dataframe per cromosoma e posizione di inizio
        df_CNV = df_CNV.sort_values(['chromosome', 'start'])
        overlapping_pairs = []
        
        # Raggruppa per cromosoma
        for chrom, group in df_CNV.groupby('chromosome'):
            genes = group.to_dict('records')
            active = []
            
            for gene in genes:
                # Rimuovi i geni che non possono più sovrapporsi
                active = [g for g in active if g['end'] >= gene['start']]
                
                # Aggiungi collegamenti per le sovrapposizioni
                for active_gene in active:
                    if active_gene['start'] <= gene['end']:
                        overlapping_pairs.append((active_gene['gene_name'], gene['gene_name']))
                
                active.append(gene)
        
        return overlapping_pairs

    @measure_time
    def create_graph(self):
        """
        Crea grafi per ogni caso nel dataset.
        """
        self.list_of_Data = []
        for case_index in range(226): 
            df_CNV = self.list_df_CNV_filled[case_index]
            
            # Crea un grafo vuoto
            G = nx.Graph()
            
            # Aggiungi i nodi con i loro attributi
            nodes_data = {row['gene_name']: {'x': row['copy_number']} 
                         for _, row in df_CNV.iterrows()}
            G.add_nodes_from(nodes_data.items())
            
            # Trova e aggiungi gli archi per i geni sovrapposti
            overlapping_pairs = self.find_overlapping_genes(df_CNV)
            G.add_edges_from(overlapping_pairs)

            # print('GRAGO N.',case_index,'\n')
            # # Numero di nodi (geni) e archi (relazioni di sovrapposizione)
            # print("Numero di nodi:", G.number_of_nodes())
            # print("Numero di archi:", G.number_of_edges())

            # # Nodo con il massimo grado (gene con più connessioni)
            # degrees = dict(G.degree())
            # max_degree_node = max(degrees, key=degrees.get)
            # print(f"Gene con il massimo grado: {max_degree_node} ({degrees[max_degree_node]} connessioni)")

            # # Trova tutte le componenti connesse
            # connected_components = list(nx.connected_components(G))
            # print("Numero di componenti connesse:", len(connected_components))
            # print('\n\n================\n\n')

            # Converti in PyTorch Geometric graph
            pyg_graph = from_networkx(G)
            pyg_graph['y'] = torch.tensor([self.os_list[case_index]])
            self.list_of_Data.append(pyg_graph)
    
    @measure_time
    def split_dataset(self):
        # First divide it in classes
        os = [int(d.y) for d in self.list_of_Data]
        os.sort()

        n = len(os)

        split_values = []
        for c in range(1, self.num_classes + 1):
            if c == self.num_classes:
                split_values.append(os[len(os) - 1])
            else:
                index = (n // self.num_classes) * c
                split_values.append(os[index - 1])

        list_data_split = []
        for c in range(self.num_classes):
            list_data_split.append([])
            for d in self.list_of_Data:
                if  (c == 0 and int(d.y) <= split_values[c]) or \
                    (c > 0 and int(d.y) <= split_values[c] and int(d.y) > split_values[c-1]):
                    d.y = torch.tensor(c)
                    list_data_split[c].append(d)
                    

        # Now split in train and test.
        self.train_list = []
        self.test_list = []

        if self.percentage_test > 0:
            test_interval = np.floor(1 / self.percentage_test)
        else:
            test_interval = len(self.list_of_Data) + 1 # we'll never reach it.
        # print(test_interval)

        for class_list in list_data_split:
            count = 1
            for d in class_list:
                if count >= test_interval:
                    self.test_list.append(d)
                    count = 0
                else:
                    self.train_list.append(d)
                count += 1

    def get_data(self):
        print("Start preprocessing", end="")
        self.preprocessing()
        print("Create the Graph", end="")
        self.create_graph()
        print("Split dataset\t", end="")
        self.split_dataset()
        
        return self.train_list, self.test_list