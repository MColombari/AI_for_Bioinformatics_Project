### Repository structure

- **Gene_sbatch**: Here we have a series of script to generate file for the training:
    - **_run.py_**: convert the Protein-to-Protein interaction file (from [link](https://string-db.org)) to ENSG format.
    - **_gene_variance.py_**: generate the score in gene variance for the provided features, the formula for the metrics can be found in the relative paper.
        - **_gene_variance_regression.py_**: same as "gene_variance.py" but for the regression training, so have different metrics.
    - **_divide_dataset.py_**: divide the user cases in train and test, and into n-classes.
        - **_divide_dataset_regression.py_**: such as "divide_dataset.py" but don't require the number of classes since there is no division in classes.
- **_edge.ipynb_** and **_NodeReduction.ipynb_**: are jupyter notebooks used for testing.
- **_Preprocessing_Gene.ipynb_**: is the actual gene preprocessing.