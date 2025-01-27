import os
import re
from datetime import datetime

class SaveModel:
    def __init__(self, main_folder_path, test_name):
        self.main_folder_path = main_folder_path

        if not os.path.isdir(main_folder_path):
            raise Exception(f"No folder with path '{main_folder_path}'")
        
        avail_dirs = [dirs for _, dirs, _ in os.walk(main_folder_path)][0]
        filtered_dir = [d for d in avail_dirs if test_name in d]

        self.current_folder = None
        if len(filtered_dir) == 0:
            # No previous test.
            index = 0
        else:
            # Find latest index.
            indexs = [int(re.findall(r'\b\d+', d)) for d in filtered_dir]
            index = max(indexs) + 1
        
        self.current_folder = f"{main_folder_path}/{test_name}_{index}"
        os.mkdir(self.current_folder)
        os.mkdir(f"{self.current_folder}/model_checkpoints")

    def save_test_info(self, more_info, start_from_checkpoint, checkpoint_path):
        with open(f"{self.current_folder}/test_info.txt", "w") as f:
            f.write("\tDate and time test start")
            f.write(f"UTC time: \"{datetime.now(datetime.timezone.utc)}\"")
            f.write(f"\tStart from checkpoint:\n{start_from_checkpoint}")
            f.write(f"\tPath start checkpoint:\n{checkpoint_path}")
            f.write(f"\tMore info:\n{more_info}")
    
    def save_model_hyperparameter(self, p_dict, model_structure):
        with open(f"{self.current_folder}/model_hyperparameter.txt", "w") as f:
            f.write("\tModel Structure")
            f.write(model_structure)
            f.write("\n\tModel hyperparameter")
            for k, v in p_dict:
                f.write(f"{k}: {v}")
