### Repository structure

- Gene_sbatch: Here we have a series of script to generate file for the training:
    - run.py: convert the Protein-to-Protein interaction file from [link](https://string-db.org) to ENSG format.
    - gene_variance.py: generate the score in gene variance for the provided feature, the formula for the metrix can be found in the relative paper.
        - gene_variance_regression.py: same as "gene_variance.py" but for the regression training, so have different metrix.
    - divide_dataset.py: provided the number of class and percentage of tran and test it divide the user case in test, train and split in each class.
        - divide_dataset_regression.py: such as "divide_dataset.py" but don't require the number of class since there is no division in classes.
- edge.ipynb and NodeReduction.ipynb: are jupyter notebooks used for testing.
- Preprocessing_Gene.ipynb: is the actual gene preprocessing.