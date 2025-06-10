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

# Here we define the class to load and preprocess data.
# So just copy the code in "Preprocessing".

class LPD:
    def __init__(self, gtf_file_path: str, folder_gene_path: str, case_id_json_path: str,
                 feature_to_save: list, feature_to_compare: str,
                 sm: SaveModel, number_of_nodes: str, variance_order_list_path: str,
                 test_file_case_id_path: str, train_file_case_id_path: str):
        self.gtf_file_path = gtf_file_path
        self.folder_gene_path = folder_gene_path
        self.case_id_json_path = case_id_json_path
        self.test_file_case_id_path = test_file_case_id_path
        self.train_file_case_id_path = train_file_case_id_path

        # All possibilitys.
        # feature_to_save = [
        #     'unstranded', 'stranded_first', 'stranded_second',
        #     'tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded'
        # ]
        # feature_to_save = ['tpm_unstranded']
        self.feature_to_save = feature_to_save

        self.feature_to_compare = feature_to_compare

        self.sm = sm

        self.number_of_nodes = number_of_nodes
        self.variance_order_list_path = variance_order_list_path

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

        self.pc_set = self.pc_set.intersection(set(list_of_nodes[:self.number_of_nodes]))

        self.sm.print(f"\n\tIntersection dim: {len(self.pc_set)}\n\t\tExecution time: ", end="")


    @measure_time
    def preprocessing(self):
        with open(self.case_id_json_path, 'r') as file:
            file_parsed = json.load(file)
        file_to_case_id = dict((file_parsed[k]['files']['gene'], k) for k in file_parsed.keys())
        file_to_os = dict((file_parsed[k]['files']['gene'], file_parsed[k]['os']) for k in file_parsed.keys())

        self.datastructure = pd.DataFrame(columns=['case_id', 'os', 'values'])

        index = 0
        # Now explore data path to get the right files
        for root, dirs, files in os.walk(self.folder_gene_path):
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
        self.THRESHOLD = 0.0004

        with open(self.test_file_case_id_path, 'r') as file:
            test_case_id = json.load(file)
        with open(self.train_file_case_id_path, 'r') as file:
            train_case_id = json.load(file)

        self.test_list = []
        self.train_list = []

        avg_edges = []
        for case_index in range(0, self.datastructure.shape[0]):
            edges = []
            in_1 = [[v] for v in list(self.datastructure['values'].loc[case_index][self.feature_to_compare])]

            dist_a = pairwise_distances(in_1, metric="euclidean")

            d_mask = np.zeros(dist_a.shape, dtype=bool)
            np.fill_diagonal(d_mask, 1)

            # Force the diagonal to be equal to Threshold, so it will not be considered, so no self loops.
            dist_a[d_mask] = self.THRESHOLD

            row, cols = np.where(dist_a < self.THRESHOLD)
            edges.append(list(row))
            edges.append(list(cols))

            avg_edges.append(len(edges[0]))

            edge_index = torch.tensor(edges, dtype=torch.long)
            x = torch.tensor(self.datastructure['values'].loc[case_index][self.feature_to_save].values, dtype=torch.float)

            if self.datastructure['case_id'].iloc[case_index] in train_case_id.keys():
                y = torch.tensor(train_case_id[self.datastructure['case_id'].iloc[case_index]])
                self.train_list.append(Data(x=x, edge_index=edge_index, y=y))
            elif self.datastructure['case_id'].iloc[case_index] in test_case_id.keys():
                y = torch.tensor(test_case_id[self.datastructure['case_id'].iloc[case_index]])
                self.test_list.append(Data(x=x, edge_index=edge_index, y=y))
            else:
                raise Exception(f"Case id not found in ether train or test\n\"{self.datastructure['case_id'].iloc[case_index]}\"")
        
        self.sm.print(f"\n\tAverage num of edges: {np.average(np.array(avg_edges))}")
        self.sm.print(f"\tVariance num of edges: {np.var(np.array(avg_edges))}")
        self.sm.print(f"\tMedian num of edges: {np.median(np.array(avg_edges))}")
        self.sm.print(f"\tMax num of edges: {max(avg_edges)}")
        self.sm.print(f"\tMin num of edges: {min(avg_edges)}\n\t\tExecution time: ", end="")


    
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

    # Here i have to split the dataset in train and test, while keeping balance
    # between all the label in each subset.
    # Return train and test separately.
    def get_data(self):
        self.sm.print("Read GTF file\t", end="")
        self.read_gtf_file()
        self.sm.print("Start preprocessing", end="")
        self.preprocessing()
        self.sm.print("Create the Graph", end="")
        self.create_graph()
        
        return self.train_list, self.test_list



class LPDEdgeKnowledgeBased(LPD):
    def __init__(self, gtf_file_path: str, folder_gene_path: str, case_id_json_path: str,
                 feature_to_save: list, feature_to_compare: str,
                 sm: SaveModel, number_of_nodes: str, variance_order_list_path: str,
                 test_file_case_id_path: str, train_file_case_id_path: str, edge_file_path: str):
        super().__init__(gtf_file_path, folder_gene_path, case_id_json_path,
                         feature_to_save, feature_to_compare,
                         sm, number_of_nodes, variance_order_list_path,
                         test_file_case_id_path, train_file_case_id_path)
        self.edge_file_path = edge_file_path

    def measure_time(func):
        def wrapper(self, *arg, **kw):
            start_time = time.time()
            ret = func(self, *arg, **kw)
            self.sm.print(f"\t\t{np.floor(time.time() - start_time)}s")
            return ret
        return wrapper

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

        index = 0
        # Now explore data path to get the right files
        for root, dirs, files in os.walk(self.folder_gene_path):
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

                            self.datastructure_gene.loc[index] = [
                                file_to_case_id[file],
                                file_to_os[file],
                                parsed_file
                            ]
                            index += 1

        # Apply log.
        for i in range(self.datastructure_gene.shape[0]):
            self.datastructure_gene['values'].loc[i][self.feature_to_save] = self.datastructure_gene['values'].loc[i][self.feature_to_save].applymap(lambda x: np.log10(x + 0.01))
        
        # Make value in a [0, 1] range.
        for r in range(self.datastructure_gene.shape[0]):
            for c in self.feature_to_save:
                self.datastructure_gene['values'].loc[r][c] =    (self.datastructure_gene['values'].loc[r][c] - self.datastructure_gene['values'].loc[r][c].min()) / \
                                                            (self.datastructure_gene['values'].loc[r][c].max() - self.datastructure_gene['values'].loc[r][c].min())
                
    @measure_time
    def preprocessing(self):
        with open(self.case_id_json_path, 'r') as file:
            file_parsed = json.load(file)
        file_to_case_id = dict((file_parsed[k]['files']['copy_number'], k) for k in file_parsed.keys())
        file_to_os = dict((file_parsed[k]['files']['copy_number'], file_parsed[k]['os']) for k in file_parsed.keys())

        self.datastructure_CNV = pd.DataFrame(columns=['case_id', 'os', 'values'])

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

                            self.datastructure_CNV.loc[index] = [
                                file_to_case_id[file],
                                file_to_os[file],
                                parsed_file
                            ]
                            index += 1

        scaler = StandardScaler()        
        for case_index in range(self.datastructure_CNV.shape[0]):
            df = self.datastructure_CNV['values'].loc[case_index]
            X_scaled = scaler.fit_transform(df['copy_number'].values.reshape(-1,1))
            df = df.drop(columns='copy_number')
            df['copy_number'] = X_scaled.flatten()
            self.list_df_CNV_filtered.append(df)
        
        ###########################################################################
        with open(PATH_FOLDER_GENE, 'r') as file:
            file_parsed = json.load(file)
        file_to_case_id = dict((file_parsed[k]['files']['gene'], k) for k in file_parsed.keys())
        file_to_os = dict((file_parsed[k]['files']['gene'], file_parsed[k]['os']) for k in file_parsed.keys())

        self.datastructure_Gene = pd.DataFrame(columns=['case_id', 'os', 'values'])

        index = 0
        # Now explore data path to get the right files
        for root, dirs, files in os.walk(PATH_FOLDER_GENE):
            for dir in dirs:
                for root, dirs, files in os.walk(PATH_FOLDER_GENE + "/" + dir):
                    for file in files:
                        if file in file_to_case_id.keys():
                            parsed_file = pd.read_csv(PATH_FOLDER_GENE + "/" + dir + "/" + file,
                                                    sep='\t', header=0, skiprows=lambda x: x in [0, 2, 3, 4, 5])
                            parsed_file = parsed_file[['gene_id'] + ['tpm_unstranded']]

                            # Now specify columns type.
                            convert_dict = dict([(k, float) for k in ['tpm_unstranded']])
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

                            self.datastructure_Gene.loc[index] = [
                                file_to_case_id[file],
                                file_to_os[file],
                                parsed_file
                            ]
                            index += 1

        # Concatenare tutti i dataframe
        df_concatenato = pd.concat(self.datastructure_Gene['values'].values)

        # Calcolare la varianza per ogni gene_id
        varianze = df_concatenato.groupby('gene_id')['copy_number'].var()

        top_n = 200  # numero di geni che si vuole mantenere
        gene_significativi = varianze.nlargest(top_n).index 

        # Apply log.
        for i in range(self.datastructure_Gene.shape[0]):
            self.datastructure_Gene['values'].loc[i][['tpm_unstranded']] = self.datastructure_Gene['values'].loc[i][self.feature_to_save].applymap(lambda x: np.log10(x + 0.01))
        
        # Make value in a [0, 1] range.
        for r in range(self.datastructure_Gene.shape[0]):
            for c in ['tpm_unstranded']:
                self.datastructure_Gene['values'].loc[r][c] =    (self.datastructure_Gene['values'].loc[r][c] - self.datastructure_Gene['values'].loc[r][c].min()) / \
                                                            (self.datastructure_Gene['values'].loc[r][c].max() - self.datastructure_Gene['values'].loc[r][c].min())
        
        for case_index in range(self.datastructure_Gene.shape[0]):
            df = self.datastructure_Gene['values'].loc[case_index][self.datastructure_Gene['values'].loc[case_index]['gene_id'].isin(gene_significativi)]
            self.list_df_Gene_filtered.append(df)

    @measure_time
    def create_graph(self):
        self.THRESHOLD = 100

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
        for case_index in range(0, self.datastructure.shape[0]):
            feature_size = self.datastructure['values'].loc[case_index][self.feature_to_compare].shape[0]
            edges = [[],[]]
            edge_attr_list = []
            miss_count = 0
            got_count = 0
            for f_1_index in range(feature_size):
                for f_2_index in range(f_1_index + 1, feature_size):
                    gene_1 = self.datastructure['values'].loc[case_index]['gene_id'].iloc[f_1_index]
                    gene_2 = self.datastructure['values'].loc[case_index]['gene_id'].iloc[f_2_index]

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

            edge_index = torch.tensor(edges, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
            x = torch.tensor(self.datastructure['values'].loc[case_index][self.feature_to_save].values, dtype=torch.float)

            if self.datastructure['case_id'].iloc[case_index] in train_case_id.keys():
                y = torch.tensor(train_case_id[self.datastructure['case_id'].iloc[case_index]])
                self.train_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            elif self.datastructure['case_id'].iloc[case_index] in test_case_id.keys():
                y = torch.tensor(test_case_id[self.datastructure['case_id'].iloc[case_index]])
                self.test_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            else:
                raise Exception(f"Case id not found in ether train or test\n\"{self.datastructure['case_id'].iloc[case_index]}\"")