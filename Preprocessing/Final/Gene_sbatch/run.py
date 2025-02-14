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

skip_first = True
with open('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/mart_export.txt', 'r') as f:
    for line in f:
        if skip_first:
            skip_first = False
            continue
        genes = line.split('\t')
        # print(genes)
        if len(genes) >= 2:
            g = genes[0]
            p = genes[1].split('\n')[0]
            conversion[p] = g

print(len(conversion.keys()))

total_row = 0
row_skipped_count = 0
with open('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/9606.protein.links.v12.0.txt', 'r') as in_file:
    with open('/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/9606.protein.links.v12.0.ENSG.txt', 'w') as out_file:
        skip_first = True
        for line in in_file:
            if skip_first:
                skip_first = False
                continue
            f, s, v = parse_row(line)
            if (not f in conversion.keys()) or (not s in conversion.keys()):
                row_skipped_count += 1
            else:
                new_f = conversion[f]
                new_s = conversion[s]
                out_file.write(f"{new_f} {new_s} {v}\n")
            total_row += 1
print(f"Total: {total_row}\tSkipped: {row_skipped_count} - {(row_skipped_count / total_row) * 100}%")