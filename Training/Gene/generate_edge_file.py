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

THRESHOLD = 300

def remove_version(x):
    if '.' in x:
        return x.split('.')[0]
    return x

# Read GTF file.
gtf = pd.read_csv('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/gencode.v47.annotation.gtf', sep="\t", header=None, comment='#')
gtf.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

parameters = ['gene_id', 'gene_type']
for p in parameters:
    gtf[p] = gtf['attribute'].apply(lambda x: re.findall(rf'{p} "([^"]*)"', x)[0] if rf'{p} "' in x else np.nan)

gtf.drop('attribute', axis=1, inplace=True)

gtf['gene_id'] = gtf['gene_id'].apply(remove_version)

gtf_pc = gtf[gtf['gene_type'] == 'protein_coding']

pc_set = set(gtf_pc['gene_id'].to_list())

accepted_gene = set()
with open('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/9606.protein.links.v12.0.ENSG.txt', 'r') as file:
    for line in file:
        # print(row_index)
        f = line.split(" ")[0]
        s = line.split(" ")[1]
        accepted_gene.add(f)
        accepted_gene.add(s)

print(f"Accepted gene dim: {len(accepted_gene)}")

pc_set = pc_set.intersection(accepted_gene)


# Load data
with open('/homes/mcolombari/AI_for_Bioinformatics_Project/Preprocessing/Final/case_id_and_structure.json', 'r') as file:
    file_parsed = json.load(file)
file_to_case_id = dict((file_parsed[k]['files']['gene'], k) for k in file_parsed.keys())
file_to_os = dict((file_parsed[k]['files']['gene'], file_parsed[k]['os']) for k in file_parsed.keys())

datastructure = pd.DataFrame(columns=['case_id', 'os', 'values'])

index = 0
# Now explore data path to get the right files
for root, dirs, files in os.walk('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression'):
    for dir in dirs:
        for root, dirs, files in os.walk('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression/' + dir):
            for file in files:
                if file in file_to_case_id.keys():
                    parsed_file = pd.read_csv('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression/' + dir + "/" + file,
                                            sep='\t', header=0, skiprows=lambda x: x in [0, 2, 3, 4, 5])
                    parsed_file = parsed_file[['gene_id']]

                    # Now specify columns type.
                    convert_dict = dict()
                    convert_dict['gene_id'] = str
                    parsed_file = parsed_file.astype(convert_dict)
                    
                    # They actually don't match.
                    # So the 'gene_type' in the dataset don't match the in the gtf file.
                    # So i'm gonna use as the right reference the gtf file.

                    parsed_file['gene_id'] = parsed_file['gene_id'].apply(remove_version)

                    # parsed_file = parsed_file[parsed_file['gene_type'] == 'protein_coding']
                    # if not set(parsed_file['gene_id']).issubset(gtf_pc_set):
                    #     raise Exception("List of coding genes don't match.")

                    parsed_file = parsed_file[parsed_file['gene_id'].isin(pc_set)]

                    datastructure.loc[index] = [
                        file_to_case_id[file],
                        file_to_os[file],
                        parsed_file
                    ]
                    index += 1

comparison_dict = {}
row_index = 0
with open('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/9606.protein.links.v12.0.ENSG.txt', 'r') as file:
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

order_of_edge = []
case_index = 0 # Just consider the first one, we don't need to check them all.
feature_size = datastructure['values'].loc[case_index]['gene_id'].shape[0]

edges = [[],[]]
miss_count = 0
got_count = 0
for f_1_index in range(feature_size):
    # Save the actual order of edges.
    order_of_edge.append(datastructure['values'].loc[case_index]['gene_id'].iloc[f_1_index])

    for f_2_index in range(f_1_index + 1, feature_size):
        gene_1 = datastructure['values'].loc[case_index]['gene_id'].iloc[f_1_index]
        gene_2 = datastructure['values'].loc[case_index]['gene_id'].iloc[f_2_index]

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
        if similarity >= THRESHOLD:
            edges[0].append(f_1_index)
            edges[0].append(f_2_index)
            edges[1].append(f_2_index)
            edges[1].append(f_1_index)

print("Similarities found")
print(f"\tMissed: {miss_count} - {(miss_count / (miss_count + got_count))*100}%")
print(f"\tGot: {got_count} - {(got_count / (miss_count + got_count))*100}%")

with open('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/edge.json', 'w') as f:
    json.dump(edges, f)

with open('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/edge_node_order.json', 'w') as f:
    json.dump(order_of_edge, f)