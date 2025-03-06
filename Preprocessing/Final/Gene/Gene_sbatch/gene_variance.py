import json
import os
import pandas as pd
import numpy as np

# PATH variable
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

index = 0
# Now explore data path to get the right files
for root, dirs, files in os.walk(PATH_FOLDER_GENE):
    for dir in dirs:
        for root, dirs, files in os.walk(PATH_FOLDER_GENE + "/" + dir):
            for file in files:
                if file in file_to_case_id.keys():
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

                    for f in feature_to_save:
                        for g in parsed_file['gene_id']:
                            if not g in gene_values_dict[f].keys():
                                gene_values_dict[f][g] = []
                            gene_values_dict[f][g].append(
                                parsed_file[parsed_file['gene_id'] == g][f])
                    index += 1

# Now mesure the variance
print("Start measuring variance")

complete_version = dict()
just_ordered = dict()
for f in feature_to_save:
    tmp = []
    for g in gene_values_dict[f].keys():
        variance = np.var(gene_values_dict[f][g])
        tmp.append((g, variance))
    tmp.sort(key=lambda x: x[1], reverse=True)
    complete_version[f] = tmp
    just_ordered[f] = [t[0] for t in tmp]

print("Save data")

with open(SAVE_FOLDER_PATH + f"/gene_variance_order_COMPLETE.json", 'w') as file:
    json.dump(complete_version, file)

for f in feature_to_save:
    with open(SAVE_FOLDER_PATH + f"/gene_variance_order_{f}.json", 'w') as file:
        json.dump(just_ordered[f], file)

print("Program End")
