### How to run the model.

The main script to run the model is train.py, here we have the complete training structure.
Then we have the following support script:
- Load_and_Process_Data.py: here we have the data loading and the preprocessing.
- models.py: all the models available.
- Save_model.py: support class to save data of the model.

config.yaml is a configuration file, here are some of the most important variable to set:
- Conditions, a series of flag use to determine if we what to save the dataset, or load from previous one.
    - Start_from_checkpoint: Start from previous checkpoint, with the same model and configuration.
    - Load_dataset: Load a dataset preivously created.
    - Save_dataset: Save the dataset if we are going to create one.
- PATH, most relevant path, usually related to the Conditions flags.
    - Test_folder_path: Where we are going to save the training results, in this forlder the program automatically generate a new folder with a increasing index.
    - Checkpoint_path: checkpoint from previous interation to start from, used if the corresponding flag is active.
    - Dataset_from_folder_path: Path of the folder where we can find the previous dataset, used if the corresponding flag is active.
- Generic_PATH: Path for dataset and other relevan files, like the json generated in the preprocessing folder.
- hyperparameter: Hyperparameter for training and dataset creation.
    - num_nodes: define the number of gene that we take from the variance file (the one with the highest score), and we use them as a white list when we create the dataset, so this doesn't represent the final number of nodes for each graph.

Then we have the test.ipynb notebook used to analyze the training results.