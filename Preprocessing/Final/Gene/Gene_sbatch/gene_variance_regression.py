import json
import os
import pandas as pd
import numpy as np

# PATH variable
PATH_TRAIN_SEPARATION = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneProcessedData/train_separation_regression.json"
GENE_ID_PROTEIN_CODING_PATH = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/gene_id_protein_coding.json"
PATH_FOLDER_GENE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression"
PATH_CASE_ID_STRUCTURE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/case_id_and_structure.json"
SAVE_FOLDER_PATH = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneProcessedData"

print("Start Code")

# Load data
with open(GENE_ID_PROTEIN_CODING_PATH, 'r') as file:
    gene_id_protein_coding_list = json.load(file)
gtf_pc_set = set(gene_id_protein_coding_list)

with open(PATH_CASE_ID_STRUCTURE, 'r') as file:
    file_parsed = json.load(file)

with open(PATH_TRAIN_SEPARATION, 'r') as file:
    train_case_id_dict = json.load(file)

NUM_CLASS = max([v for v in train_case_id_dict.values()]) + 1

file_to_case_id = dict((file_parsed[k]['files']['gene'], k) for k in file_parsed.keys())
file_to_os = dict((file_parsed[k]['files']['gene'], file_parsed[k]['os']) for k in file_parsed.keys())

def remove_version(x):
    if '.' in x:
        return x.split('.')[0]
    return x

# All possibilitys.
# feature_to_save = [
#     'unstranded', 'stranded_first', 'stranded_second',
#     'tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded'
# ]
feature_to_save = ['tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded']

gene_values_dict = dict()
for f in feature_to_save:
    gene_values_dict[f] = dict()


print("Start reading")
gene_list = []
index = 0
gene_list_flag = True
# Now explore data path to get the right files
for root, dirs, files in os.walk(PATH_FOLDER_GENE):
    for dir in dirs:
        for root, dirs, files in os.walk(PATH_FOLDER_GENE + "/" + dir):
            for file in files:
                if file in file_to_case_id.keys():
                    case_id = file_to_case_id[file]
                    if not case_id in train_case_id_dict.keys():
                        continue

                    parsed_file = pd.read_csv(PATH_FOLDER_GENE + "/" + dir + "/" + file,
                                              sep='\t', header=0, skiprows=lambda x: x in [0, 2, 3, 4, 5])
                    parsed_file = parsed_file[['gene_id'] + feature_to_save]
                    # parsed_file = parsed_file[['gene_id', 'gene_type'] + feature_to_save]

                    # Now specify columns type.
                    convert_dict = dict([(k, float) for k in feature_to_save])
                    convert_dict['gene_id'] = str
                    parsed_file = parsed_file.astype(convert_dict)
                    
                    # They actually don't match.
                    # So the 'gene_type' in the dataset don't match the in the gtf file.
                    # So i'm gonna use as the right reference the gtf file.

                    parsed_file['gene_id'] = parsed_file['gene_id'].apply(remove_version)

                
                    # parsed_file = parsed_file[parsed_file['gene_type'] == 'protein_coding']
                    # if not set(parsed_file['gene_id']).issubset(gtf_pc_set):
                    #     raise Exception("List of coding genes don't match.")

                    parsed_file = parsed_file[parsed_file['gene_id'].isin(gtf_pc_set)]

                    # Remove gene duplicate if found.
                    if not parsed_file['gene_id'].is_unique:
                        print("Duplicate found")
                        print(f"\tBefore drop duplicate: {parsed_file.shape[0]} Row")
                        parsed_file = parsed_file.groupby('gene_id', as_index=False).mean()
                        print(f"\tAfter drop duplicate: {parsed_file.shape[0]} Row")

                    for f in feature_to_save:
                        for g in parsed_file['gene_id']:
                            if gene_list_flag:
                                gene_list.append(g)
                            else:
                                assert g in gene_list
                            if not g in gene_values_dict[f].keys():
                                gene_values_dict[f][g] = []

                            value = parsed_file.loc[parsed_file['gene_id'] == g, f].values[0]
                            log_value = np.log10(value + 0.01)
                            gene_values_dict[f][g].append(log_value)
                        gene_list_flag = False
                    index += 1

print(f"Size gene_list {len(gene_list)}")

# Now mesure the variance
print("Start measuring variance")

intra_variance = {}
for f in feature_to_save:
    intra_variance[f] = dict()
    for g in gene_list:
        tmp = 0
        assert g in gene_values_dict[f].keys()
        variance = np.var(gene_values_dict[f][g])
        intra_variance[f][g] = variance

score = dict()
ordered_gene = dict()
for f in feature_to_save:
    tmp = []
    for g in gene_list:
        s = intra_variance[f][g]
        tmp.append((g, s))
    tmp.sort(key=lambda x: x[1], reverse=True)
    score[f] = tmp
    ordered_gene[f] = [t[0] for t in tmp]

print("Save data")

with open(SAVE_FOLDER_PATH + f"/gene_variance_order_COMPLETE_regression.json", 'w') as file:
    json.dump(score, file)

for f in feature_to_save:
    with open(SAVE_FOLDER_PATH + f"/gene_variance_order_{f}_regression.json", 'w') as file:
        json.dump(ordered_gene[f], file)
    print(f"{f}:\t{len(score[f])} Entry")
    

print("Program End")
