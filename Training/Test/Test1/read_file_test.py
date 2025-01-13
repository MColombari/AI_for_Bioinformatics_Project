import torch
from torch_geometric.data import Data
import json
from pathlib import Path
import os
import pandas as pd

PATH = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression/0b52237e-f467-420a-b7b2-952c0004efc9/d90d2b4f-8865-440d-a8b7-e0b7c947e095.rna_seq.augmented_star_gene_counts.tsv"

parsed_file = pd.read_csv(PATH, sep='\t', header=0, skiprows=lambda x: x in [0, 2, 3, 4, 5])
print(type(parsed_file))
print(parsed_file.dtypes)
print(parsed_file.size)
print(parsed_file.shape)
print(parsed_file.columns)
print(parsed_file)
print(list(parsed_file["gene_id"]))
# print(parsed_file.loc[1:])
# print(parsed_file[4,:]['tpm_unstranded'])
