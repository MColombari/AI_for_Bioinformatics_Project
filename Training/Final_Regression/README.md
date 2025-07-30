### How to run the model.

The main script to run the model is train.py, here we have the complete training structure.
Then we have the following support script:
- **_Load_and_Process_Data.py_**: here we have the data loading and the preprocessing.
- **_models.py_**: all the models available.
- **_Save_model.py_**: support class to save data of the model.

**_config.yaml_** is a configuration file, here are some of the most important variable to set:
- **_Conditions_**, a series of boolean values used as flags for the script:
    - _Start_from_checkpoint_: Start from previous checkpoint, with the same model and configuration.
    - _Load_dataset_: Load a dataset previously created.
    - _Save_dataset_: Save the dataset if we are going to create one.
    - _Dont_train_: Typically used to just create the dataset and don't actually train any model.
- **_PATH_**: most relevant paths, usually related to the Conditions flags.
    - _Test_folder_path_: Where we are going to save the training results, in this folder the program automatically generates a new folder with an increasing index for each training.
    - _Checkpoint_path_: checkpoint from previous iteration to start from, used if the corresponding flag is active.
    - _Dataset_from_folder_path_: Path of the folder where we can find the previous dataset, used if the corresponding flag is active.
- **_Generic_PATH_**: Paths for dataset and other relevant files, like the json generated in the preprocessing folder.
- **_hyperparameter_**: Hyperparameter for training and dataset creation.
    - _num_nodes_: define the number of genes that we take from the variance file (the one with the highest score), and we use them as a white list when we create the dataset, so this doesn't represent the final number of nodes for each graph.

Then we have the **_test.ipynb_** notebook used to analyze the training results.