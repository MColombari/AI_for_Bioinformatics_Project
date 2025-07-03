import pandas as pd
import numpy as np
import json
import os
import time


PATH_FOLDER_GENE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneExpression"
PATH_CASE_ID_STRUCTURE = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/case_id_and_structure.json"
PATH_OUTPUT_FOLDER = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/GeneProcessedData"
NUM_CLASSES = 64
PERCENTAGE_TEST = 0.15

class LPD_modified:
    def __init__(self, folder_gene_path: str, case_id_json_path: str,
                 num_classes: int, percentage_test: float,
                 ):
        self.folder_gene_path = folder_gene_path
        self.case_id_json_path = case_id_json_path

        self.num_classes = num_classes
        self.percentage_test = percentage_test

    def measure_time(func):
        def wrapper(self, *arg, **kw):
            start_time = time.time()
            ret = func(self, *arg, **kw)
            print(f"\t\t{np.floor(time.time() - start_time)}s")
            return ret
        return wrapper


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
                            parsed_file = parsed_file[['gene_id']]

                            # Now specify columns type.
                            convert_dict = dict()
                            convert_dict['gene_id'] = str
                            parsed_file = parsed_file.astype(convert_dict)

                            self.datastructure.loc[index] = [
                                file_to_case_id[file],
                                file_to_os[file],
                                parsed_file
                            ]
                            index += 1


    @measure_time
    def split_dataset(self):
        self.datastructure = self.datastructure.sort_values(by=['os'])
        os = self.datastructure["os"].to_list()

        n = len(os)

        split_values = []
        for c in range(1, self.num_classes + 1):
            if c == self.num_classes:
                split_values.append(os[-1])
            else:
                index = (n // self.num_classes) * c
                split_values.append(os[index - 1])

        list_data_split = []
        for c in range(self.num_classes):
            list_data_split.append({})
            for _, row in self.datastructure.iterrows():
                if  (c == 0 and int(row['os']) <= split_values[c]) or \
                    (c > 0 and int(row['os']) <= split_values[c] and int(row['os']) > split_values[c-1]):
                    row['os'] = c
                    list_data_split[c][row['case_id']] = c

        print(f"\n\tNum of instances per class:")
        for c in range(self.num_classes):
            print(f"\t\tClass {c} -> {len(list_data_split[c])} case.")

        # Now split in train and test.
        self.train_list = {}
        self.test_list = {}

        if self.percentage_test > 0:
            test_interval = np.floor(1 / self.percentage_test)
        else:
            test_interval = self.datastructure.shape[0] + 1 # we'll never reach it.
        # print(test_interval)

        for class_list in list_data_split:
            count = 1
            for k in class_list.keys():
                if count >= test_interval:
                    self.test_list[k] = class_list[k]
                    count = 0
                else:
                    self.train_list[k] = class_list[k]
                count += 1

        print(f"\tTrain size: {len(self.train_list)}")
        print(f"\tTest size: {len(self.test_list)}")

        print(f"\t\t\tExecution time: ", end="")

    # Here i have to split the dataset in train and test, while keeping balance
    # between all the label in each subset.
    # Return train and test separately.
    def get_data(self):
        print("Start preprocessing", end="")
        self.preprocessing()
        print("Split dataset\t", end="")
        self.split_dataset()
        
        return self.train_list, self.test_list
    

lpd = LPD_modified(PATH_FOLDER_GENE, PATH_CASE_ID_STRUCTURE, NUM_CLASSES, PERCENTAGE_TEST)

train_out, test_out = lpd.get_data()

with open(f'{PATH_OUTPUT_FOLDER}/train_separation_{NUM_CLASSES}_classes.json', 'w') as f:
    json.dump(train_out, f)

with open(f'{PATH_OUTPUT_FOLDER}/test_separation_{NUM_CLASSES}_classes.json', 'w') as f:
    json.dump(test_out, f)