# Convert ENSP to ENSG.

# ESNG represents a gene – a segment of DNA that contains the instructions for making proteins (or other functional RNA).
# ESNP represents a protein – the product of a gene that has been transcribed into mRNA and translated into a protein.

def parse_row(row):
    first = row.split(".")[1].split(" ")[0]
    second = row.split(".")[2].split(" ")[0]
    val = int(row.split(" ")[-1])

    return first, second, val

esnp_set = set()

# Read file
with open('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/9606.protein.links.v12.0.txt', 'r') as file:
    skip_first = True
    for line in file:
        if skip_first:
            skip_first = False
            continue
        f, s, v = parse_row(line)
        esnp_set.add(f)
        esnp_set.add(s)


# Ensembl REST API

import requests

conversion = {}

ensp_ids = list(esnp_set)
server = "https://rest.ensembl.org"
ext = "/lookup/id/"

headers = {"Content-Type": "application/json"}

found_count = 0
not_found_count = 0
for ensp in ensp_ids:
    response = requests.get(server + ext + ensp, headers=headers)
    if response.ok:
        data = response.json()
        conversion[ensp] = str(data.get('gene_id'))
        found_count += 1
    else:
        conversion[ensp] = None
        not_found_count += 1

print(f"Found {found_count}\t Not Found {not_found_count} - {not_found_count / (found_count + not_found_count)}%")

row_skipped_count = 0
with open('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/9606.protein.links.v12.0.txt', 'r') as in_file:
    with open('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/9606.protein.links.v12.0.ENSG.txt', 'w') as out_file:
        skip_first = True
        for line in in_file:
            if skip_first:
                skip_first = False
                continue
            f, s, v = parse_row(line)
            new_f = conversion[f]
            new_s = conversion[s]
            if new_f and new_s:
                out_file.write(f"{new_f} {new_s} {v}\n")
            else:
                row_skipped_count += 1
print(f"{row_skipped_count} row skipped")