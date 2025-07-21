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
from Save_model import SaveModel
import random


class LPDEdgeKnowledgeBased:
    def __init__(self, gtf_file_path: str, folder_gene_path: str, folder_methylation_path:str,
                 folder_copy_number_path:str, case_id_json_path: str, methylation_converter_file_path:str,
                 test_file_case_id_path: str, train_file_case_id_path: str, edge_file_path: str,
                 variance_order_list_path: str,
                 feature_to_save_dict: dict, number_of_nodes: str,
                 sm: SaveModel):
        
        # Path variable
        self.gtf_file_path = gtf_file_path
        self.folder_gene_path = folder_gene_path
        self.folder_methylation_path = folder_methylation_path
        self.folder_copy_number_path = folder_copy_number_path
        self.case_id_json_path = case_id_json_path
        self.methylation_converter_file_path = methylation_converter_file_path
        self.test_file_case_id_path = test_file_case_id_path
        self.train_file_case_id_path = train_file_case_id_path
        self.edge_file_path = edge_file_path
        self.variance_order_list_path = variance_order_list_path

        # Data to create dataset
        self.feature_to_save_dict = feature_to_save_dict
        self.number_of_nodes = number_of_nodes

        # Extra.
        self.sm = sm


    def measure_time(func):
        def wrapper(self, *arg, **kw):
            start_time = time.time()
            ret = func(self, *arg, **kw)
            self.sm.print(f"\t\t{np.floor(time.time() - start_time)}s")
            return ret
        return wrapper
    
    def remove_version(self, x):
        if '.' in x:
            return x.split('.')[0]
        return x

    @measure_time
    def read_gtf_file(self):
        # We read the GTF file and the edge file, to keep only the gene that we have there.
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
        self.sm.print(f"\n\tProtein coding dim: {len(self.pc_set)}")

        accepted_gene = set()
        with open(self.edge_file_path, 'r') as file:
            for line in file:
                # print(row_index)
                f = line.split(" ")[0]
                s = line.split(" ")[1]
                accepted_gene.add(f)
                accepted_gene.add(s)

        self.sm.print(f"\tAccepted gene dim: {len(accepted_gene)}")

        self.pc_set = self.pc_set.intersection(accepted_gene)

        self.sm.print(f"\tIntersection with accepted gene dim: {len(self.pc_set)}")

        # Take only the first n nodes in order of variance.
        with open(self.variance_order_list_path, 'r') as file:
            list_of_nodes = json.load(file)

        self.pc_set = self.pc_set.intersection(set(list_of_nodes[:self.number_of_nodes]))

        self.sm.print(f"\tIntersection dim: {len(self.pc_set)}\n\t\tExecution time: ", end="")


    @measure_time
    def preprocessing_gene(self):
        with open(self.case_id_json_path, 'r') as file:
            file_parsed = json.load(file)
        file_to_case_id = dict((file_parsed[k]['files']['gene'], k) for k in file_parsed.keys())
        file_to_os = dict((file_parsed[k]['files']['gene'], file_parsed[k]['os']) for k in file_parsed.keys())

        self.datastructure_gene = pd.DataFrame(columns=['case_id', 'os', 'values'])

        feature_to_save = self.feature_to_save_dict['gene']

        index = 0
        # Now explore data path to get the right files
        for root, dirs, files in os.walk(self.folder_gene_path):
            for dir in dirs:
                for root, dirs, files in os.walk(self.folder_gene_path + "/" + dir):
                    for file in files:
                        if file in file_to_case_id.keys():
                            parsed_file = pd.read_csv(self.folder_gene_path + "/" + dir + "/" + file,
                                                    sep='\t', header=0, skiprows=lambda x: x in [0, 2, 3, 4, 5])
                            parsed_file = parsed_file[['gene_id'] + feature_to_save]

                            # Now specify columns type.
                            convert_dict = dict([(k, float) for k in feature_to_save])
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

                            self.datastructure_gene.loc[index] = [
                                file_to_case_id[file],
                                file_to_os[file],
                                parsed_file
                            ]
                            index += 1

        # Apply log.
        for i in range(self.datastructure_gene.shape[0]):
            self.datastructure_gene['values'].loc[i][feature_to_save] = self.datastructure_gene['values'].loc[i][feature_to_save].applymap(lambda x: np.log10(x + 0.01))
        
        # Make value in a [0, 1] range.
        for r in range(self.datastructure_gene.shape[0]):
            for c in feature_to_save:
                self.datastructure_gene['values'].loc[r][c] =    (self.datastructure_gene['values'].loc[r][c] - self.datastructure_gene['values'].loc[r][c].min()) / \
                                                                (self.datastructure_gene['values'].loc[r][c].max() - self.datastructure_gene['values'].loc[r][c].min())

    def convert_methylation_to_gene(self, methylation_id, conversion_dict):
        return conversion_dict.get(methylation_id, None)

    def preprocessing_methylation(self):
        # Load the file path dictionary
        with open(self.case_id_json_path, 'r') as file:
            file_parsed = json.load(file)

        # Create dictionaries for case_id and os
        file_to_case_id = {file_parsed[k]['files']['methylation']: k for k in file_parsed.keys()}
        file_to_os = {file_parsed[k]['files']['methylation']: file_parsed[k]['os'] for k in file_parsed.keys()}

        # Initialize the DataFrame
        self.datastructure_methylation = pd.DataFrame(columns=['case_id', 'os','values'])

        feature_to_save = self.feature_to_save_dict['methylation']

        index = 0
        for root, dirs, files in os.walk(self.folder_methylation_path):
            for dir in dirs:
                for root, dirs, files in os.walk(os.path.join(self.folder_methylation_path, dir)):
                    for file in files:
                        if file in file_to_case_id.keys():
                            parsed_file = pd.read_csv(os.path.join(self.folder_methylation_path, dir, file),
                                                      sep='\t', header=None, names=["id", "methylation"])

                            convert_dict = dict([(k, float) for k in feature_to_save])
                            convert_dict['id'] = str
                            parsed_file = parsed_file.astype(convert_dict)

                            parsed_file = parsed_file.dropna()

                            # Extract methylation values
                            # methylation_id = parsed_file['id'].tolist()
                            # methylation_values = parsed_file['methylation'].tolist()

                            # Add the data to the DataFrame
                            self.datastructure_methylation.loc[index] = [
                                file_to_case_id[file],
                                file_to_os[file],
                                parsed_file
                            ]
                            index += 1

        # Carica il file di conversione
        conversion_df = pd.read_csv(self.methylation_converter_file_path, dtype = {'gene_id': str, 'gene_chr': str, 'gene_strand': str, 'gene_start': str, 'gene_end': str, 'cpg_island': str, 'cpg_IlmnID': str, 'cpg_chr': str})
        # Crea un dizionario per la conversione rapida
        conversion_dict = pd.Series(conversion_df.gene_id.values, index=conversion_df.cpg_IlmnID).to_dict()
        # Crea una nuova colonna 'gene_id' nel DataFrame
        number_of_duplicate_list = []
        for i in range(self.datastructure_methylation.shape[0]):
            self.datastructure_methylation['values'].iloc[i]['gene_id'] = self.datastructure_methylation['values'].iloc[i]['id'].apply(lambda x: self.convert_methylation_to_gene(x, conversion_dict))
            self.datastructure_methylation.at[i, 'values'] = self.datastructure_methylation.at[i, 'values'].drop(columns=['id'])
            self.datastructure_methylation.at[i, 'values'] = self.datastructure_methylation.at[i, 'values'][
                self.datastructure_methylation.at[i, 'values']['gene_id'].isin(self.pc_set)
            ]
            
            # self.sm.print(f"\t\t\tNumber of total gene: {len([v for v in self.datastructure_methylation['values'].loc[i]['gene_id'].duplicated()])}")
            # self.sm.print(f"\t\t\tNumber of duplicate: {len([v for v in self.datastructure_methylation['values'].loc[i]['gene_id'].duplicated() if v == True])}")
            number_of_duplicate_list.append(len([v for v in self.datastructure_methylation['values'].loc[i]['gene_id'].duplicated() if v == True]))
            self.datastructure_methylation.at[i, 'values'] = self.datastructure_methylation.at[i, 'values'].drop_duplicates(subset=['gene_id'])
            assert self.datastructure_methylation['values'].loc[i]['gene_id'].duplicated().any() == False
        self.sm.print("")
        self.sm.print("\t\tNumber of duplicate gene:")
        self.sm.print(f"\t\t\tmin: {min(number_of_duplicate_list)}")
        self.sm.print(f"\t\t\tmax: {max(number_of_duplicate_list)}")
        self.sm.print(f"\t\t\tavg: {0 if len(number_of_duplicate_list) == 0 else sum(number_of_duplicate_list)/len(number_of_duplicate_list)}")

        # Make value in a [0, 1] range.
        for r in range(self.datastructure_methylation.shape[0]):
            for c in feature_to_save:
                self.datastructure_methylation['values'].loc[r][c] =   (self.datastructure_methylation['values'].loc[r][c] - self.datastructure_methylation['values'].loc[r][c].min()) / \
                                                                (self.datastructure_methylation['values'].loc[r][c].max() - self.datastructure_methylation['values'].loc[r][c].min())        


    @measure_time
    def preprocessing_copy_number(self):
        with open(self.case_id_json_path, 'r') as file:
            file_parsed = json.load(file)
        file_to_case_id = dict((file_parsed[k]['files']['copy_number'], k) for k in file_parsed.keys())
        file_to_os = dict((file_parsed[k]['files']['copy_number'], file_parsed[k]['os']) for k in file_parsed.keys())

        self.datastructure_copy_number = pd.DataFrame(columns=['case_id', 'os', 'values'])

        feature_to_save = self.feature_to_save_dict['copy_number']

        index = 0
        # Now explore data path to get the right files
        for root, dirs, files in os.walk(self.folder_copy_number_path):
            for dir in dirs:
                for root, dirs, files in os.walk(self.folder_copy_number_path + "/" + dir):
                    for file in files:
                        if file in file_to_case_id.keys():
                            parsed_file = pd.read_csv(self.folder_copy_number_path + "/" + dir + "/" + file, sep='\t')
                            parsed_file = parsed_file[['gene_id'] + feature_to_save]

                            # Now specify columns type.
                            convert_dict = dict([(k, float) for k in feature_to_save])
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

                            self.datastructure_copy_number.loc[index] = [
                                file_to_case_id[file],
                                file_to_os[file],
                                parsed_file
                            ]
                            index += 1

        # Make value in a [0, 1] range.
        for r in range(self.datastructure_copy_number.shape[0]):
            for c in feature_to_save:
                self.datastructure_copy_number['values'].loc[r][c] =    (self.datastructure_copy_number['values'].loc[r][c] - self.datastructure_copy_number['values'].loc[r][c].min()) / \
                                                                        (self.datastructure_copy_number['values'].loc[r][c].max() - self.datastructure_copy_number['values'].loc[r][c].min())


    @measure_time
    def datastructure_merge_func(self):
        self.datastructure_merge = pd.DataFrame(columns=['case_id', 'os', 'values'])

        Number_of_miss_case_id_methylation = 0
        Number_of_miss_case_id_copy_number = 0
        Number_of_miss_on_both = 0

        merge_index = 0

        final_number_of_node = []
        for index in range(self.datastructure_gene.shape[0]):
            curr_case_id = self.datastructure_gene['case_id'].loc[index]
            curr_gene_datastructure = self.datastructure_gene[self.datastructure_gene['case_id'] == curr_case_id]
            curr_methylation_datastructure = self.datastructure_methylation[self.datastructure_methylation['case_id'] == curr_case_id]
            curr_copy_number_datastructure = self.datastructure_copy_number[self.datastructure_copy_number['case_id'] == curr_case_id]

            assert curr_gene_datastructure.shape[0] == 1
            assert curr_methylation_datastructure.shape[0] <= 1
            assert curr_copy_number_datastructure.shape[0] <= 1

            if curr_methylation_datastructure.shape[0] == 0:
                if curr_copy_number_datastructure.shape[0] == 0:
                    Number_of_miss_on_both += 1
                    continue
                else:
                    Number_of_miss_case_id_methylation += 1
            if curr_copy_number_datastructure.shape[0] == 0:
                Number_of_miss_case_id_copy_number += 1

            # self.sm.print(f"\t\t\tShape of gene_datastructure before merge: {curr_gene_datastructure['values'].iloc[0].shape[0]}")

            merged_value =  curr_gene_datastructure['values'].iloc[0].merge(curr_copy_number_datastructure['values'].iloc[0], on='gene_id', how='inner')
            # self.sm.print(f"\t\t\tShape of merged_value after copy number merge: {merged_value.shape[0]}")

            merged_value = merged_value.merge(curr_methylation_datastructure['values'].iloc[0], on='gene_id', how='inner')
            # self.sm.print(f"\t\t\tShape of merged_value after Methylation merge: {merged_value.shape[0]}\n")
            final_number_of_node.append(merged_value.shape[0])
                                                                    

            self.datastructure_merge.loc[merge_index] = [
                curr_case_id,
                curr_gene_datastructure['os'],
                merged_value
            ]
            merge_index += 1

        self.sm.print("")
        self.sm.print("\t\tNumber of final node:")
        self.sm.print(f"\t\t\tmin: {min(final_number_of_node)}")
        self.sm.print(f"\t\t\tmax: {max(final_number_of_node)}")
        self.sm.print(f"\t\t\tavg: {0 if len(final_number_of_node) == 0 else sum(final_number_of_node)/len(final_number_of_node)}")

        self.sm.print(f"\t\tNumber of case_id miss due to methylation: {Number_of_miss_case_id_methylation}")
        self.sm.print(f"\t\tNumber of case_id miss due to copy number: {Number_of_miss_case_id_copy_number}")
        self.sm.print(f"\t\tNumber of case_id miss due to both: {Number_of_miss_case_id_copy_number}")

    @measure_time
    def create_graph(self):
        self.THRESHOLD = 400

        comparison_dict = {}
        row_index = 0
        with open(self.edge_file_path, 'r') as file:
            for line in file:
                row_index += 1
                # print(row_index)
                f = line.split(" ")[0]
                s = line.split(" ")[1]
                v = int(line.split(" ")[2].split('\n')[0])
                k = [f, s]
                k.sort()
                k = tuple(k)
                if k in comparison_dict.keys():
                    old_v = comparison_dict[k]
                    if old_v < v:
                        comparison_dict[k] = v
                else:
                    comparison_dict[k] = v

        with open(self.test_file_case_id_path, 'r') as file:
            test_case_id = json.load(file)
        with open(self.train_file_case_id_path, 'r') as file:
            train_case_id = json.load(file)

        self.test_list = []
        self.train_list = []
        for case_index in range(0, self.datastructure_merge.shape[0]):
            feature_size = self.datastructure_merge['values'].loc[case_index]['gene_id'].shape[0]
            edges = [[],[]]
            edge_attr_list = []
            miss_count = 0
            got_count = 0
            for f_1_index in range(feature_size):
                for f_2_index in range(f_1_index + 1, feature_size):
                    gene_1 = self.datastructure_merge['values'].loc[case_index]['gene_id'].iloc[f_1_index]
                    gene_2 = self.datastructure_merge['values'].loc[case_index]['gene_id'].iloc[f_2_index]

                    # print(gene_1)
                    # print(gene_2)
                    k = [gene_1, gene_2]
                    k.sort()
                    k = tuple(k)
                    # print(k)

                    if k in comparison_dict.keys():
                        similarity = comparison_dict[k]
                        got_count += 1
                        # print(f"Got it\t{similarity}")
                    else:
                        similarity = 0
                        miss_count += 1
                        # print("Drop it")
                        # print("Similarity not found")
                    
                    # In this case the higher the number the more similarity.
                    if similarity >= self.THRESHOLD:
                        edges[0].append(f_1_index)
                        edges[0].append(f_2_index)
                        edges[1].append(f_2_index)
                        edges[1].append(f_1_index)
                        # I need to append two attribute because we have 2 edge, one in one direction, and the
                        # second in the opposite direction.
                        edge_attr_list.append([similarity])
                        edge_attr_list.append([similarity])

            print("Similarities found")
            print(f"\tMissed: {miss_count} - {(miss_count / (miss_count + got_count))*100}%")
            print(f"\tGot: {got_count} - {(got_count / (miss_count + got_count))*100}%")

            feature_to_save = []
            for k in self.feature_to_save_dict.keys():
                for e in self.feature_to_save_dict[k]:
                    feature_to_save.append(e)

            edge_index = torch.tensor(edges, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
            x = torch.tensor(self.datastructure_merge['values'].loc[case_index][feature_to_save].values, dtype=torch.float)

            if self.datastructure_merge['case_id'].iloc[case_index] in train_case_id.keys():
                y = torch.tensor(train_case_id[self.datastructure_merge['case_id'].iloc[case_index]], dtype=torch.float)
                self.train_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            elif self.datastructure_merge['case_id'].iloc[case_index] in test_case_id.keys():
                y = torch.tensor(test_case_id[self.datastructure_merge['case_id'].iloc[case_index]], dtype=torch.float)
                self.test_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            else:
                raise Exception(f"Case id not found in ether train or test\n\"{self.datastructure_merge['case_id'].iloc[case_index]}\"")

    def get_data(self):
        self.sm.print("Read GTF file\t", end="")
        self.read_gtf_file()
        self.sm.print("Start preprocessing Methylation", end="")
        self.preprocessing_methylation()
        self.sm.print("Start preprocessing Copy Number", end="")
        self.preprocessing_copy_number()
        self.sm.print("Start preprocessing Gene", end="")
        self.preprocessing_gene()
        self.sm.print("Start merge", end="")
        self.datastructure_merge_func()
        self.sm.print("Create the Graph", end="")
        self.create_graph()
        
        return self.train_list, self.test_list