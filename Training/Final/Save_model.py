import os
import re
from datetime import datetime, timezone
import torch

class SaveModel:
    def __init__(self, main_folder_path, test_name):
        self.main_folder_path = main_folder_path
        self.test_name = test_name

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
            indexs = [int(d.split("_")[-1]) for d in filtered_dir]
            index = max(indexs) + 1
        
        self.current_folder = f"{main_folder_path}/{test_name}_{index}"
        os.mkdir(self.current_folder)
        os.mkdir(f"{self.current_folder}/model_checkpoints")

    def save_test_info(self, more_info, start_from_checkpoint, checkpoint_path):
        with open(f"{self.current_folder}/test_info.txt", "w") as f:
            f.write("Date and time test start\n")
            f.write(f"\tUTC time: \"{datetime.now(timezone.utc)}\"\n")
            f.write(f"Start from checkpoint:\n\t{start_from_checkpoint}\n")
            f.write(f"Path start checkpoint:\n\t{checkpoint_path}\n")
            f.write(f"More info:\n\t{more_info}\n")
    
    def save_model_hyperparameter(self, p_dict):
        with open(f"{self.current_folder}/model_hyperparameter.txt", "w") as f:
            f.write("\tModel hyperparameter\n")
            for k in p_dict.keys():
                f.write(f"{k}: {p_dict[k]}\n")

    def save_model_architecture(self, model):
        with open(f"{self.current_folder}/model_architecture.txt", "w") as f:
            f.write("\tModel Structure")
            f.write(str(model))

    def save_epoch(self, epoch, model):
        torch.save({
            'epoch': epoch,
            'model_dict':model.state_dict()}, 
            f"{self.current_folder}/model_checkpoints/{self.test_name}_epoch_{epoch}.pth")
        
    def save_dataset(self, train_dataset, test_dataset):
        os.mkdir(f"{self.current_folder}/datasets")
        torch.save(train_dataset, f"{self.current_folder}/datasets/train.pkl")
        torch.save(test_dataset, f"{self.current_folder}/datasets/test.pkl")

    # logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score*100))
    def save_epoch_data(self, epoch, loss_train, accuracy_train, loss_test, accuracy_test):
        with open(f"{self.current_folder}/epoch_data.txt", "a") as f:
            f.write(f"Epoch: '{epoch}', Loss Train: '{loss_train}', Accuracy Train: '{accuracy_train}', Loss Test: '{loss_test}', Accuracy Test: '{accuracy_test}'\n")
    
    def print(self, string:str, end:str="\n"):
        with open(f"{self.current_folder}/model_prints.txt", "a") as f:
            f.write(string + end)
        print(string, end=end)

class SaveTest:
    def __init__(self, main_folder_path, test_name):
        self.main_folder_path = main_folder_path
        self.test_name = test_name

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
            indexs = [int(d.split("_")[-1]) for d in filtered_dir]
            index = max(indexs) + 1
        
        self.current_folder = f"{main_folder_path}/{test_name}_{index}"
        os.mkdir(self.current_folder)

    def save_results(self, res: list):
        with open(f"{self.current_folder}/results.txt", "w") as f:
            for row in res:
                f.write(f"{row}\n")
