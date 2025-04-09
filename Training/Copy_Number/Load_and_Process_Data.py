import pandas as pd
import re
import numpy as np
import json
import torch
import os
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import time
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Here we define the class to load and preprocess data.
# So just copy the code in "Preprocessing".

def measure_time(func):
    def wrapper(x):
        start_time = time.time()
        func(x)
        print(f"\t\t{np.floor(time.time() - start_time)}s")
    return wrapper

class LPD:
    def __init__(self, gtf_file_path: str, folder_gene_path: str, case_id_json_path: str, top_n: int,
                 feature_to_save: list, feature_to_compare: str, num_classes: int, percentage_test: float
                 ):
        # PATH_GTF_FILE = "/homes/mcolombari/AI_for_Bioinformatics_Project/Personal/gencode.v47.annotation.gtf"
        self.gtf_file_path = gtf_file_path
        # PATH_FOLDER_GENE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression"
        self.folder_gene_path = folder_gene_path
        # PATH_CASE_ID_STRUCTURE = "./case_id_and_structure.json"
        self.case_id_json_path = case_id_json_path

        self.feature_to_save = feature_to_save
        self.feature_to_compare = feature_to_compare
        self.list_df_CNV_filtered = []
        self.top_n = top_n

        # NUMBER_OF_CLASSES = 3
        self.num_classes = num_classes
        # PERCENTAGE_OF_TEST = 0.3
        self.percentage_test = percentage_test

    def remove_version(self, x):
        if '.' in x:
            return x.split('.')[0]
        return x


    @measure_time
    def read_gtf_file(self):
        gtf = pd.read_csv(self.gtf_file_path, sep="\t", header=None, comment='#')
        gtf.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

        parameters = ['gene_id', 'gene_type']
        for p in parameters:
            gtf[p] = gtf['attribute'].apply(lambda x: re.findall(rf'{p} "([^"]*)"', x)[0] if rf'{p} "' in x else np.nan)

        gtf.drop('attribute', axis=1, inplace=True)

        gtf['gene_id'] = gtf['gene_id'].apply(self.remove_version)

        gtf_pc = gtf[gtf['gene_type'] == 'protein_coding']

        # Protein coding set
        self.pc_set = set(gtf_pc['gene_id'].to_list())


    @measure_time
    def preprocessing(self):
        with open(self.case_id_json_path, 'r') as file:
            file_parsed = json.load(file)
        file_to_case_id = dict((file_parsed[k]['files']['copy_number'], k) for k in file_parsed.keys())
        file_to_os = dict((file_parsed[k]['files']['copy_number'], file_parsed[k]['os']) for k in file_parsed.keys())

        self.datastructure = pd.DataFrame(columns=['case_id', 'os', 'values'])

        index = 0
        # Now explore data path to get the right files
        for root, dirs, files in os.walk(self.folder_gene_path):
            for dir in dirs:
                for root, dirs, files in os.walk(self.folder_gene_path + "/" + dir):
                    for file in files:
                        if file in file_to_case_id.keys():
                            parsed_file = pd.read_csv(self.folder_gene_path + "/" + dir + "/" + file, sep='\t')
                            parsed_file = parsed_file[['gene_id'] + self.feature_to_save]

                            # Now specify columns type.
                            convert_dict = dict([(k, float) for k in self.feature_to_save])
                            convert_dict['gene_id'] = str
                            parsed_file = parsed_file.astype(convert_dict)
                            
                            # They actually don't match.
                            # So the 'gene_type' in the dataset don't match the in the gtf file.
                            # So i'm gonna use as the right reference the gtf file.

                            parsed_file['gene_id'] = parsed_file['gene_id'].apply(self.remove_version)

                            # parsed_file = parsed_file[parsed_file['gene_type'] == 'protein_coding']
                            # if not set(parsed_file['gene_id']).issubset(gtf_pc_set):
                            #     raise Exception("List of coding genes don't match.")

                            parsed_file = parsed_file[parsed_file['gene_id'].isin(self.pc_set)].fillna(0)

                            self.datastructure.loc[index] = [
                                file_to_case_id[file],
                                file_to_os[file],
                                parsed_file
                            ]
                            index += 1
        
        # Concatenare tutti i dataframe
        df_concatenato = pd.concat(self.datastructure['values'].values)

        # Calcolare la varianza per ogni gene_id
        varianze = df_concatenato.groupby('gene_id')['copy_number'].var()

        gene_significativi = varianze.nlargest(self.top_n).index 

        scaler = StandardScaler()
                
        for case_index in range(self.datastructure.shape[0]):
            df = self.datastructure['values'].loc[case_index][self.datastructure['values'].loc[case_index]['gene_id'].isin(gene_significativi)]
            X_scaled = scaler.fit_transform(df['copy_number'].values.reshape(-1,1))
            df = df.drop(columns='copy_number')
            df['copy_number'] = X_scaled.flatten()
            self.list_df_CNV_filtered.append(df)


    @measure_time
    def create_graph(self):
        self.THRESHOLD = 0.06

        self.list_of_Data = []
        for case_index in range(0, self.datastructure.shape[0]):
            edges = []
            in_1 = [[v] for v in list(self.list_df_CNV_filtered[case_index][self.feature_to_compare])]

            dist_a = pairwise_distances(in_1, metric="euclidean")

            d_mask = np.zeros(dist_a.shape, dtype=bool)
            np.fill_diagonal(d_mask, 1)

            # Force the diagonal to be equal to Threshold, so it will not be considered, so no self loops.
            dist_a[d_mask] = self.THRESHOLD

            row, cols = np.where(dist_a < self.THRESHOLD)
            edges.append(list(row))
            edges.append(list(cols))

            edge_index = torch.tensor(edges, dtype=torch.long)
            x = torch.tensor(list(self.list_df_CNV_filtered[case_index][self.feature_to_compare]), dtype=torch.float)
            y = torch.tensor(self.datastructure['os'].loc[case_index])
            self.list_of_Data.append(Data(x=x, edge_index=edge_index, y=y))

            # G = to_networkx(Data(x=x, edge_index=edge_index, y=y), to_undirected=True)
            # print(f"\n\n### Graph {case_index} ###")
            # print("Numero di nodi:", G.number_of_nodes())
            # print("Numero di edge:", G.number_of_edges())
            # degrees = dict(G.degree())
            # # Nodo con il massimo grado (gene con più connessioni)
            # max_degree_node = max(degrees, key=degrees.get)
            # print(f"Nodo con il massimo grado: {max_degree_node} ({degrees[max_degree_node]} connessioni)")

            # # Trova tutte le componenti connesse
            # connected_components = list(nx.connected_components(G))
            # print("Numero di componenti connesse: ", len(connected_components))
            # print(f"density: {nx.density(G)}")

    
    def get_instance_class(self, d):
        # So we give in unput an instance of data and get the respenctive data
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

        for c in range(self.num_classes):
            if  (c == 0 and int(d.y) <= split_values[c]) or \
                (c > 0 and int(d.y) <= split_values[c] and int(d.y) > split_values[c-1]):
                return c
        raise Exception("No Class found")

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

    # Here i have to split the dataset in train and test, while keeping balance
    # between all the label in each subset.
    # Return train and test separately.
    def get_data(self):
        print("Read GTF file\t", end="")
        self.read_gtf_file()
        print("Start preprocessing", end="")
        self.preprocessing()
        print("Create the Graph", end="")
        self.create_graph()
        print("Split dataset\t", end="")
        self.split_dataset()
        
        return self.train_list, self.test_list
