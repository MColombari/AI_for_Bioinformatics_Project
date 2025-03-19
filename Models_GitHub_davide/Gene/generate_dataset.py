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
import random

# Here we define the class to load and preprocess data.
# So just copy the code in "Preprocessing".

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('num_nodes', type=int, help='A required integer positional argument')
parser.add_argument('threshold', type=float, help='A required float positional argument')

args = parser.parse_args()

class ModifiesLPD:
    def __init__(self, gtf_file_path: str, folder_gene_path: str, case_id_json_path: str,
                 feature_to_save: list, feature_to_compare: str, num_classes: int,
                 number_of_nodes: str, variance_order_list_path: str, threshold: int):
        self.gtf_file_path = gtf_file_path
        self.folder_gene_path = folder_gene_path
        self.case_id_json_path = case_id_json_path

        # All possibilitys.
        # feature_to_save = [
        #     'unstranded', 'stranded_first', 'stranded_second',
        #     'tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded'
        # ]
        # feature_to_save = ['tpm_unstranded']
        self.feature_to_save = feature_to_save

        self.feature_to_compare = feature_to_compare

        self.num_classes = num_classes

        self.number_of_nodes = number_of_nodes
        self.variance_order_list_path = variance_order_list_path

        self.THRESHOLD = threshold

        t_value = str(self.THRESHOLD).split('.')[-1]
        self.MAIN_DIR = f'../datasets/GENE_EXP_{self.number_of_nodes}_{t_value}'
        if not os.path.exists(self.MAIN_DIR):
            os.makedirs(self.MAIN_DIR)

        if os.path.exists(f'{self.MAIN_DIR}/GENE_EXP_A.txt'):
            os.remove(f'{self.MAIN_DIR}/GENE_EXP_A.txt')
        if os.path.exists(f'{self.MAIN_DIR}/GENE_EXP_graph_indicator.txt'):
            os.remove(f'{self.MAIN_DIR}/GENE_EXP_graph_indicator.txt')
        if os.path.exists(f'{self.MAIN_DIR}/GENE_EXP_graph_attributes.txt'):
            os.remove(f'{self.MAIN_DIR}/GENE_EXP_graph_attributes.txt')
        if os.path.exists(f'{self.MAIN_DIR}/GENE_EXP_graph_labels.txt'):
            os.remove(f'{self.MAIN_DIR}/GENE_EXP_graph_labels.txt')

    def measure_time(func):
        def wrapper(self, *arg, **kw):
            start_time = time.time()
            ret = func(self, *arg, **kw)
            print(f"\t\t{np.floor(time.time() - start_time)}s")
            return ret
        return wrapper

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

        # Take only the first n nodes in order of variance.
        with open(self.variance_order_list_path, 'r') as file:
            list_of_nodes = json.load(file)

        # Dont work we need to make sure to have a fidex number of nodes. TODO
        self.pc_set = self.pc_set.intersection(set(list_of_nodes[:self.number_of_nodes]))

        print(f"\n\tIntersection dim: {len(self.pc_set)}\n\t\tExecution time: ", end="")


    @measure_time
    def preprocessing(self):
        with open(self.case_id_json_path, 'r') as file:
            file_parsed = json.load(file)
        file_to_case_id = dict((file_parsed[k]['files']['gene'], k) for k in file_parsed.keys())
        file_to_os = dict((file_parsed[k]['files']['gene'], file_parsed[k]['os']) for k in file_parsed.keys())

        self.datastructure = pd.DataFrame(columns=['case_id', 'os', 'values'])

        index = 0
        # Now explore data path to get the right files
        for _, dirs, files in os.walk(self.folder_gene_path):
            for dir in dirs:
                for root, dirs, files in os.walk(self.folder_gene_path + "/" + dir):
                    for file in files:
                        if file in file_to_case_id.keys():
                            parsed_file = pd.read_csv(self.folder_gene_path + "/" + dir + "/" + file,
                                                    sep='\t', header=0, skiprows=lambda x: x in [0, 2, 3, 4, 5])
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

                            parsed_file = parsed_file[parsed_file['gene_id'].isin(self.pc_set)]

                            parsed_file.drop_duplicates(subset=['gene_id'])
                            # for i in list(parsed_file['gene_id']):
                            #     print(i)
                            # raise Exception("Stop")

                            self.datastructure.loc[index] = [
                                file_to_case_id[file],
                                file_to_os[file],
                                parsed_file
                            ]
                            index += 1

        # Apply log.
        for i in range(self.datastructure.shape[0]):
            self.datastructure['values'].loc[i][self.feature_to_save] = self.datastructure['values'].loc[i][self.feature_to_save].applymap(lambda x: np.log10(x + 0.01))
        
        # Make value in a [0, 1] range.
        for r in range(self.datastructure.shape[0]):
            for c in self.feature_to_save:
                self.datastructure['values'].loc[r][c] =    (self.datastructure['values'].loc[r][c] - self.datastructure['values'].loc[r][c].min()) / \
                                                            (self.datastructure['values'].loc[r][c].max() - self.datastructure['values'].loc[r][c].min())
                

    @measure_time
    def create_graph(self):    
        self.list_of_os = []
        avg_edges = []
        offset = 0
        for case_index in range(self.datastructure.shape[0]):
            in_1 = [[v] for v in list(self.datastructure['values'].loc[case_index][self.feature_to_compare])]

            dist_a = pairwise_distances(in_1, metric="euclidean")

            d_mask = np.zeros(dist_a.shape, dtype=bool)
            np.fill_diagonal(d_mask, 1)

            # Force the diagonal to be equal to Threshold, so it will not be considered, so no self loops.
            dist_a[d_mask] = self.THRESHOLD

            row, cols = np.where(dist_a < self.THRESHOLD)

            avg_edges.append(len(row))

            # print(self.datastructure['values'].loc[case_index][self.feature_to_save].shape)
            # print(self.datastructure['values'].loc[case_index][self.feature_to_save].values)

            self.list_of_os.append(self.datastructure['os'].iloc[case_index])

            # Write edges.
            with open(f'{self.MAIN_DIR}/GENE_EXP_A.txt','a') as file:
                for i in range(len(row)):
                    file.write(f'{(row[i] + 1) + offset}, {(cols[i] + 1) + offset}\n')
            
            # Write edges index.
            with open(f'{self.MAIN_DIR}/GENE_EXP_graph_indicator.txt','a') as file:
                for i in range(len(in_1)):
                    file.write(f'{case_index + 1}\n')

            # Write edges index.
            with open(f'{self.MAIN_DIR}/GENE_EXP_graph_attributes.txt','a') as file:
                for n in self.datastructure['values'].loc[case_index][self.feature_to_save].values:
                    val = [str(v) for v in n]
                    file.write(', '.join(val) + '\n')
            
            offset += len(in_1)

        
        print(f"\n\tAverage num of edges: {np.average(np.array(avg_edges))}")
        print(f"\tVariance num of edges: {np.var(np.array(avg_edges))}")
        print(f"\tMedian num of edges: {np.median(np.array(avg_edges))}")
        print(f"\tMax num of edges: {max(avg_edges)}")
        print(f"\tMin num of edges: {min(avg_edges)}\n\t\tExecution time: ", end="")


    @measure_time
    def write_graph_label(self):
        os = [int(d) for d in self.list_of_os]
        os.sort()

        n = len(os)

        split_values = []
        for c in range(1, self.num_classes + 1):
            if c == self.num_classes:
                split_values.append(os[-1])
            else:
                index = (n // self.num_classes) * c
                split_values.append(os[index - 1])
        
        # Write edges index.
        with open(f'{self.MAIN_DIR}/GENE_EXP_graph_labels.txt','w') as file:
            for o in self.list_of_os:
                for c in range(self.num_classes):
                    if  (c == 0 and o <= split_values[c]) or \
                        (c > 0 and o <= split_values[c] and o > split_values[c-1]):
                        file.write(f'{c + 1}\n')

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
        print("Start graph labeling", end="")
        self.write_graph_label()
    

# Load data path
PATH_GTF_FILE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/gencode.v47.annotation.gtf"
PATH_FOLDER_GENE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression"
PATH_CASE_ID_STRUCTURE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/case_id_and_structure.json"
# For edge similarity files.
PATH_EDGE_FILE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/9606.protein.links.v12.0.ENSG.txt"
PATH_COMPLETE_EDGE_FILE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/edge_T900.json"
PATH_EDGE_ORDER_FILE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/edge_node_order.json"
# Order of nodes files.
PATH_ORDER_GENE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneProcessedData/gene_variance_order_tpm_unstranded.json"

hyperparameter = {
    'num_classes': 2,
    'num_nodes': args.num_nodes,
    'feature_to_save': ['tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded'], # Specify parameter for gene.
    'feature_to_compare': 'tpm_unstranded'
}


lpd = ModifiesLPD(PATH_GTF_FILE, PATH_FOLDER_GENE, PATH_CASE_ID_STRUCTURE,
                            hyperparameter['feature_to_save'], hyperparameter['feature_to_compare'],
                            hyperparameter['num_classes'],
                            hyperparameter['num_nodes'], PATH_ORDER_GENE, args.threshold)
lpd.get_data()  # List of Data.

print("Program terminated successfully")
