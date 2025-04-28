# Here we define the class to load and preprocess data.
# So just copy the code in "Preprocessing".
import pandas as pd
from matplotlib import pyplot as plt
import os
import json
import seaborn as sns
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from matplotlib import pyplot as plt
import torch
import os
import numpy as np
from sklearn.metrics import pairwise_distances
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

PATH_METHYLATION = "/work/h2020deciderficarra_shared/TCGA/OV/project_n16_data/Methylation"
FILE_PATH_DICT = "/homes/fmancinelli/progettoBio/AI_for_Bioinformatics_Project/Preprocessing/Final/case_id_and_structure.json"
FILE_PATH_CONVERTER = "/homes/fmancinelli/progettoBio/AI_for_Bioinformatics_Project/Preprocessing/Final/Methylation/matched_cpg_genes.csv"
FILE_PATH_DATASTRUCTURE_CONVERTED = 'datastructure_converted.csv'
NUMBER_OF_VALUES = 1000
NUMBER_OF_CLASSES = 2
PERCENTAGE_TEST = 0.2

class LPD:
    

    def __init__(self):
        # Load the file path dictionary
        with open(FILE_PATH_DICT, 'r') as file:
            file_parsed = json.load(file)

        # Create dictionaries for case_id and os
        file_to_case_id = {file_parsed[k]['files']['methylation']: k for k in file_parsed.keys()}
        file_to_os = {file_parsed[k]['files']['methylation']: file_parsed[k]['os'] for k in file_parsed.keys()}

        # Initialize the DataFrame
        datastructure = pd.DataFrame(columns=['case_id', 'os','methylation_id','methylation_values'])

        index = 0
        for root, dirs, files in os.walk(PATH_METHYLATION):
            for dir in dirs:
                for root, dirs, files in os.walk(os.path.join(PATH_METHYLATION, dir)):
                    for file in files:
                        if file in file_to_case_id.keys():
                            parsed_file = pd.read_csv(os.path.join(PATH_METHYLATION, dir, file),
                                                      sep='\t', header=None, names=["id", "methylation"])
                            parsed_file = parsed_file[['id', 'methylation']]
                            parsed_file = parsed_file.astype({'methylation': float, 'id': str})

                            # Extract methylation values
                            methylation_id = parsed_file['id'].tolist()
                            methylation_values = parsed_file['methylation'].tolist()

                            # Add the data to the DataFrame
                            datastructure.loc[index] = [
                                file_to_case_id[file],
                                file_to_os[file],
                                methylation_id,
                                methylation_values
                            ]
                            index += 1

        # Carica il file di conversione
        conversion_df = pd.read_csv(FILE_PATH_CONVERTER, dtype = {'gene_id': str, 'gene_chr': str, 'gene_strand': str, 'gene_start': str, 'gene_end': str, 'cpg_island': str, 'cpg_IlmnID': str, 'cpg_chr': str})
        # Crea un dizionario per la conversione rapida
        conversion_dict = pd.Series(conversion_df.gene_id.values, index=conversion_df.cpg_IlmnID).to_dict()

        # Crea una nuova colonna 'gene_id' nel DataFrame
        datastructure['gene_id'] = datastructure['methylation_id'].apply(lambda x: self.convert_methylation_to_gene(x, conversion_dict))

        # Calcolo della varianza per ogni lista in methylation_values
        datastructure['methylation_variance'] = datastructure['methylation_values'].apply(lambda x: pd.Series(x).var())

        # Trova le prime mille posizioni dei valori con la varianza più alta
        max_variance_positions = datastructure['methylation_values'].apply(lambda x: pd.Series(x).nlargest(10000).index.tolist())

        # Creazione di un nuovo DataFrame con solo i valori nelle posizioni elencate
        df_variance = {
            'case_id': datastructure['case_id'],
            'os': datastructure['os'],
            'methylation_id': datastructure.apply(lambda row: [row['methylation_id'][i] for i in max_variance_positions[row.name]], axis=1),
            'methylation_values': datastructure.apply(lambda row: [row['methylation_values'][i] for i in max_variance_positions[row.name]], axis=1),
            'gene_id': datastructure.apply(lambda row: [row['gene_id'][i] for i in max_variance_positions[row.name]], axis=1),
            'methylation_variance': datastructure['methylation_variance']  # Aggiungi methylation_variance al DataFrame
        }

        df_variance = pd.DataFrame(df_variance)

        # Ordinamento del DataFrame in base alla varianza in ordine decrescente
        df_variance = df_variance.sort_values(by='methylation_variance', ascending=False)

        #print("Nuovo DataFrame con solo i valori nelle posizioni elencate:")
        #print(df_variance)
        #print(df_variance.info())
        # Applica la funzione al DataFrame
        df_variance = self.remove_none_and_corresponding_positions(df_variance)

        # Rimuovi le righe dove la lista gene_id è vuota dopo aver rimosso i valori None
        df_variance = df_variance[df_variance['gene_id'].map(len) > 0]

        #print("Nuovo DataFrame con solo i valori nelle posizioni elencate:")
        #print(df_variance)
        # Creazione di un nuovo DataFrame con i 2000 valori con varianza più alta da df_variance
        top_data = {
            'case_id': df_variance['case_id'],
            'os': df_variance['os'],
            'methylation_id': df_variance['methylation_id'].apply(lambda x: x[:NUMBER_OF_VALUES]),
            'methylation_values': df_variance['methylation_values'].apply(lambda x: x[:NUMBER_OF_VALUES]),
            'gene_id': df_variance['gene_id'].apply(lambda x: x[:NUMBER_OF_VALUES]),
            'methylation_variance': df_variance['methylation_variance']  # Mantieni methylation_variance nel nuovo DataFrame
        }

        self.df = pd.DataFrame(top_data)

        #print("Nuovo DataFrame con i 2000 valori con varianza più alta:")
        #print(self.df)
        # Ordina il DataFrame per varianza in ordine decrescente
        df_sorted = self.df.sort_values(by='methylation_variance', ascending=False)

        # Estrai la colonna gene_id in ordine ordinato
        sorted_gene_ids = df_sorted['gene_id'].tolist()

        # Salva i gene_id ordinati in un file JSON
        with open('sorted_gene_ids_methylation.json', 'w') as f:
            json.dump(sorted_gene_ids, f)

        #print("Gene IDs ordinati salvati in sorted_gene_ids_methylation.json")
        self.df.to_csv(FILE_PATH_DATASTRUCTURE_CONVERTED, index=False)

        # Creiamo una lista per contenere tutti i valori di metilazione
        all_methylation_values = []

        # Converte la colonna in un tensor PyTorch
        os_tensor = torch.tensor(self.df['os'].values, dtype=torch.float)

        # Itera attraverso il DataFrame e aggiungi i valori di metilazione alla lista
        for index, row in self.df.iterrows():
            methylation_values = row['methylation_values']
            all_methylation_values.extend(methylation_values)

        # Calcola la media e la mediana
        #mean_methylation = pd.Series(all_methylation_values).mean()
        median_methylation = pd.Series(all_methylation_values).median()

        #print("Media della metilazione:", mean_methylation)
        #print("Mediana della metilazione:", median_methylation)

        THRESHOLD = median_methylation

        self.list_of_Data = []
        methylation_data = self.df['methylation_values'].values
        feature_size = methylation_data.shape[0]
        edges = [[], []]

        # Calcola la matrice delle distanze
        dist_matrix = np.zeros((feature_size, feature_size))
        for i in range(feature_size):
            for j in range(i + 1, feature_size):
                dist_matrix[i, j] = np.linalg.norm(np.array(methylation_data[i]) - np.array(methylation_data[j]))

        # Trova gli indici dove la similarità è inferiore o uguale alla soglia
        f_1_indices, f_2_indices = np.where(dist_matrix <= THRESHOLD)
        for f_1_index, f_2_index in zip(f_1_indices, f_2_indices):
            if f_1_index < f_2_index:  # Assicurati di non duplicare gli indici
                edges[0].append(f_1_index)
                edges[0].append(f_2_index)
                edges[1].append(f_2_index)
                edges[1].append(f_1_index)

        # Converti le liste di valori in array numpy e poi in tensori torch
        methylation_data = np.array([np.array(x) for x in methylation_data], dtype=np.float32)
        x = torch.tensor(methylation_data, dtype=torch.float)
        # Calcola il valore minimo e massimo del tensor
        x_min = torch.min(x)
        x_max = torch.max(x)

        # Applica la normalizzazione
        x_normalized = (x - x_min) / (x_max - x_min)
        print('x: ')
        print(x_normalized)
        print('y: ')
        print(os_tensor)
        print('edges: ')
        print(edges)
        edge_index = torch.tensor(edges, dtype=torch.long)
        self.list_of_Data.append(Data(x=x_normalized, edge_index=edge_index, y=os_tensor))
        print('list of data: ')
        print(self.list_of_Data)
        #G1 = to_networkx(self.list_of_Data[0], to_undirected=True)

        #print("Numero di nodi:", G1.number_of_nodes())
        #print("Numero di archi:", G1.number_of_edges())

        # Grado di ciascun nodo (numero di connessioni per gene)
        #degrees = dict(G1.degree())
        #print("Gradi dei nodi:", degrees)

        # Nodo con il massimo grado (gene con più connessioni)
        #max_degree_node = max(degrees, key=degrees.get)
        #print(f"Gene con il massimo grado: {max_degree_node} ({degrees[max_degree_node]} connessioni)")

        # Trova tutte le componenti connesse
        #connected_components = list(nx.connected_components(G1))
        #print("Numero di componenti connesse:", len(connected_components))

    # Funzione per rimuovere i valori None dalle liste e le posizioni corrispondenti in altre colonne
    def remove_none_and_corresponding_positions(self, df):
        # Trova le posizioni dei valori None in gene_id
        none_positions = df['gene_id'].apply(lambda x: [i for i, value in enumerate(x) if value is None])

        # Rimuovi i valori None da gene_id e le posizioni corrispondenti da methylation_id e methylation_values
        df['gene_id'] = df['gene_id'].apply(lambda x: [value for value in x if value is not None])
        df['methylation_id'] = df.apply(lambda row: [value for i, value in enumerate(row['methylation_id']) if i not in none_positions[row.name]], axis=1)
        df['methylation_values'] = df.apply(lambda row: [value for i, value in enumerate(row['methylation_values']) if i not in none_positions[row.name]], axis=1)

        return df
    
    # Funzione per convertire methylation_id in gene_id utilizzando il dizionario
    def convert_methylation_to_gene(self, methylation_ids, conversion_dict):
        return [conversion_dict.get(methylation_id, None) for methylation_id in methylation_ids]

    # Here i have to split the dataset in train and test, while keeping balance
    # between all the label in each subset.
    # Return train and test separately.
    def get_data(self):
        # Ottenere e ordinare tutti i valori di 'y'
        os = []
        print(f"Numero di dati in self.list_of_Data: {len(self.list_of_Data)}")
        for d in self.list_of_Data:
            print(d)
        for d in self.list_of_Data:
            if isinstance(d.y, torch.Tensor):
                os.extend(d.y.view(-1).tolist())  # Converte i tensori multidimensionali in lista
            else:
                raise ValueError(f"d.y non è un tensore valido: {d.y}")

        # Ordina i valori
        os.sort()
        n = len(os)

        # Calcola i valori di split
        split_values = []
        for c in range(1, NUMBER_OF_CLASSES + 1):
            if c == NUMBER_OF_CLASSES:
                split_values.append(os[-1])  # Ultimo valore per l'ultima classe
            else:
                index = (n // NUMBER_OF_CLASSES) * c
                split_values.append(os[index - 1])

        # Assegna ogni valore a una classe
        for d in self.list_of_Data:
            if isinstance(d.y, torch.Tensor):
                new_y = []
                for value in d.y.view(-1).tolist():  # Itera sui valori di d.y
                    for c in range(NUMBER_OF_CLASSES):
                        if (c == 0 and value <= split_values[c]) or \
                           (c > 0 and value <= split_values[c] and value > split_values[c - 1]):
                            new_y.append(c)  # Assegna la classe corrispondente
                d.y = torch.tensor(new_y)  # Assegna il nuovo tensore con le classi

        print(f"Numero di classi: {NUMBER_OF_CLASSES}")
        print(f"Valori di split: {split_values}")
        for d in self.list_of_Data:
            print(f"y originale: {d.y}")
            print(f"y riassegnato: {d.y.view(-1).tolist()}")
        # Mescola i dati e suddividi in training e test set
        random.shuffle(self.list_of_Data)
        total = len(self.list_of_Data)
        n_test = max(1, int(np.floor(total * PERCENTAGE_TEST)))  # Assicura almeno 1 elemento nel test set
        self.test_list = self.list_of_Data[:n_test]
        self.train_list = self.list_of_Data[n_test:]
        print(f"Totale dati: {total}")
        print(f"Dati nel test set: {n_test}")
        print("train list: ")
        print(self.train_list)
        print("test list: ")
        print(self.test_list)
        return self.train_list, self.test_list