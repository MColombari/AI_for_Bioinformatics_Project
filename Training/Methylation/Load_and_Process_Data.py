# Here we define the class to load and preprocess data.
# Versione corretta con tutti gli errori risolti
import pandas as pd
from matplotlib import pyplot as plt
import os
import json
import seaborn as sns
import networkx as nx
from torch_geometric.data import Data
from matplotlib import pyplot as plt
import torch
import os
import numpy as np
import random
import pandas as pd
from torch_geometric.data import Data, Batch


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
            'methylation_variance': datastructure['methylation_variance']
        }

        df_variance = pd.DataFrame(df_variance)

        # Ordinamento del DataFrame in base alla varianza in ordine decrescente
        df_variance = df_variance.sort_values(by='methylation_variance', ascending=False)

        # Applica la funzione al DataFrame
        df_variance = self.remove_none_and_corresponding_positions(df_variance)

        # Rimuovi le righe dove la lista gene_id è vuota dopo aver rimosso i valori None
        df_variance = df_variance[df_variance['gene_id'].map(len) > 0]

        # Creazione di un nuovo DataFrame con i NUMBER_OF_VALUES valori con varianza più alta
        top_data = {
            'case_id': df_variance['case_id'],
            'os': df_variance['os'],
            'methylation_id': df_variance['methylation_id'].apply(lambda x: x[:NUMBER_OF_VALUES]),
            'methylation_values': df_variance['methylation_values'].apply(lambda x: x[:NUMBER_OF_VALUES]),
            'gene_id': df_variance['gene_id'].apply(lambda x: x[:NUMBER_OF_VALUES]),
            'methylation_variance': df_variance['methylation_variance']
        }

        self.df = pd.DataFrame(top_data)

        # Ordina il DataFrame per varianza in ordine decrescente
        df_sorted = self.df.sort_values(by='methylation_variance', ascending=False)

        # Estrai la colonna gene_id in ordine ordinato
        sorted_gene_ids = df_sorted['gene_id'].tolist()

        # Salva i gene_id ordinati in un file JSON
        with open('sorted_gene_ids_methylation.json', 'w') as f:
            json.dump(sorted_gene_ids, f)

        self.df.to_csv(FILE_PATH_DATASTRUCTURE_CONVERTED, index=False)

        #Creazione dei grafi
        self.list_of_Data = []
        methylation_data = self.df['methylation_values'].values
        feature_size = methylation_data.shape[0]

        # Converti le liste di valori in array numpy
        methylation_data_arrays = []
        for i in range(feature_size):
            # Filtra i valori NaN
            values = [v for v in methylation_data[i] if not pd.isna(v)]
            if len(values) > 0:
                methylation_data_arrays.append(np.array(values, dtype=np.float32))
            else:
                print(f"Attenzione: campione {i} non ha valori validi")
                continue

        # Calcola statistiche per normalizzazione globale
        all_values = []
        for arr in methylation_data_arrays:
            all_values.extend(arr.tolist())
        
        if len(all_values) == 0:
            raise ValueError("Nessun valore di metilazione valido trovato")

        # Calcola mediana per la soglia di connessione
        median_methylation = np.median(all_values)
        global_min = min(all_values)
        global_max = max(all_values)
        
        print(f"Mediana della metilazione: {median_methylation}")
        print(f"Range valori: [{global_min}, {global_max}]")

        # Converti la colonna os in un tensor PyTorch
        os_values = self.df['os'].values[:len(methylation_data_arrays)]
        os_tensor = torch.tensor(os_values, dtype=torch.float)

        # Crea i grafi corretti
        for i, sample_values in enumerate(methylation_data_arrays):
            if len(sample_values) == 0:
                continue
                
            # Normalizza i valori del campione
            if global_max > global_min:
                normalized_values = (sample_values - global_min) / (global_max - global_min)
            else:
                normalized_values = sample_values
            
            num_nodes = len(normalized_values)
            
            # Crea le connessioni basate sulla similarità dei valori
            edges = [[], []]
            threshold = 0.1  # Soglia di similarità
            
            for node_i in range(num_nodes):
                for node_j in range(node_i + 1, num_nodes):
                    # Calcola la distanza tra i valori di metilazione
                    dist = abs(normalized_values[node_i] - normalized_values[node_j])
                    if dist <= threshold:
                        edges[0].extend([node_i, node_j])
                        edges[1].extend([node_j, node_i])
            
            # Se non ci sono edge, crea almeno alcune connessioni per evitare nodi isolati
            if len(edges[0]) == 0:
                # Connetti ogni nodo al suo più vicino
                for node_i in range(num_nodes):
                    if node_i < num_nodes - 1:
                        edges[0].extend([node_i, node_i + 1])
                        edges[1].extend([node_i + 1, node_i])
            
            # Crea tensori
            edge_index = torch.tensor(edges, dtype=torch.long)
            node_features = torch.tensor(normalized_values, dtype=torch.float).unsqueeze(1)  # Shape: [num_nodes, 1]
            
            # Verifica che gli indici degli edge siano validi
            if edge_index.numel() > 0:
                max_edge_idx = edge_index.max().item()
                if max_edge_idx >= num_nodes:
                    print(f"Errore nel campione {i}: max edge index {max_edge_idx} >= num_nodes {num_nodes}")
                    continue
            
            self.list_of_Data.append(Data(
                x=node_features,
                edge_index=edge_index,
                y=os_tensor[i]
            ))

        print(f"Numero totale di grafi creati: {len(self.list_of_Data)}")
        
        # Stampa statistiche di debug
        if len(self.list_of_Data) > 0:
            print("Statistiche dei grafi creati:")
            node_counts = [data.x.shape[0] for data in self.list_of_Data]
            edge_counts = [data.edge_index.shape[1] for data in self.list_of_Data]
            print(f"  Nodi per grafo - min: {min(node_counts)}, max: {max(node_counts)}, media: {np.mean(node_counts):.2f}")
            print(f"  Edge per grafo - min: {min(edge_counts)}, max: {max(edge_counts)}, media: {np.mean(edge_counts):.2f}")
            
            # Verifica alcuni campioni
            for i in range(min(3, len(self.list_of_Data))):
                data = self.list_of_Data[i]
                print(f"  Campione {i}: {data.x.shape[0]} nodi, {data.edge_index.shape[1]} edge, y={data.y.item():.4f}")

    def remove_none_and_corresponding_positions(self, df):
        """Funzione per rimuovere i valori None dalle liste e le posizioni corrispondenti in altre colonne"""
        # Trova le posizioni dei valori None in gene_id
        none_positions = df['gene_id'].apply(lambda x: [i for i, value in enumerate(x) if value is None])

        # Rimuovi i valori None da gene_id e le posizioni corrispondenti da methylation_id e methylation_values
        df['gene_id'] = df['gene_id'].apply(lambda x: [value for value in x if value is not None])
        df['methylation_id'] = df.apply(lambda row: [value for i, value in enumerate(row['methylation_id']) if i not in none_positions[row.name]], axis=1)
        df['methylation_values'] = df.apply(lambda row: [value for i, value in enumerate(row['methylation_values']) if i not in none_positions[row.name]], axis=1)

        return df
    
    def convert_methylation_to_gene(self, methylation_ids, conversion_dict):
        """Funzione per convertire methylation_id in gene_id utilizzando il dizionario"""
        return [conversion_dict.get(methylation_id, None) for methylation_id in methylation_ids]

    def get_data(self):
        """
        Split del dataset in train e test, mantenendo il bilanciamento
        tra tutte le label in ogni subset.
        Return train e test separatamente.
        """
        if len(self.list_of_Data) == 0:
            raise ValueError("Nessun dato disponibile per il training")
        
        # Estrai tutti i valori di 'y'
        y_values = []
        for data in self.list_of_Data:
            if isinstance(data.y, torch.Tensor):
                if data.y.numel() == 1:
                    y_values.append(data.y.item())
                else:
                    # Se y ha più elementi, prendi il primo
                    y_values.append(data.y.flatten()[0].item())
            else:
                y_values.append(float(data.y))

        # Converti in classificazione binaria
        y_values = np.array(y_values)
        
        # Usa la mediana come soglia per la classificazione binaria
        threshold = np.median(y_values)
        print(f"Soglia per classificazione binaria: {threshold}")
        
        # Aggiorna i valori y nei Data objects
        for i, data in enumerate(self.list_of_Data):
            # Converti a classe binaria (0 o 1)
            binary_class = 1 if y_values[i] > threshold else 0
            data.y = torch.tensor(binary_class, dtype=torch.long)

        # Crea liste per ciascuna classe per bilanciare il dataset
        class_0_data = [data for data in self.list_of_Data if data.y.item() == 0]
        class_1_data = [data for data in self.list_of_Data if data.y.item() == 1]
        
        print(f"Distribuzione classi: Classe 0: {len(class_0_data)}, Classe 1: {len(class_1_data)}")
        
        # Mescola i dati all'interno di ogni classe
        random.shuffle(class_0_data)
        random.shuffle(class_1_data)
        
        # Calcola il numero di campioni per il test set per ogni classe
        n_test_class_0 = max(1, int(len(class_0_data) * PERCENTAGE_TEST))
        n_test_class_1 = max(1, int(len(class_1_data) * PERCENTAGE_TEST))
        
        # Split bilanciato
        test_class_0 = class_0_data[:n_test_class_0]
        train_class_0 = class_0_data[n_test_class_0:]
        
        test_class_1 = class_1_data[:n_test_class_1]
        train_class_1 = class_1_data[n_test_class_1:]
        
        # Combina le classi
        self.test_list = test_class_0 + test_class_1
        self.train_list = train_class_0 + train_class_1
        
        # Mescola i dataset finali
        random.shuffle(self.test_list)
        random.shuffle(self.train_list)
        
        print(f"Dataset finale:")
        print(f"  Training set: {len(self.train_list)} campioni")
        print(f"  Test set: {len(self.test_list)} campioni")
        
        # Verifica distribuzione finale
        train_class_0_count = sum(1 for data in self.train_list if data.y.item() == 0)
        train_class_1_count = sum(1 for data in self.train_list if data.y.item() == 1)
        test_class_0_count = sum(1 for data in self.test_list if data.y.item() == 0)
        test_class_1_count = sum(1 for data in self.test_list if data.y.item() == 1)
        
        print(f"  Training - Classe 0: {train_class_0_count}, Classe 1: {train_class_1_count}")
        print(f"  Test - Classe 0: {test_class_0_count}, Classe 1: {test_class_1_count}")
        
        return self.train_list, self.test_list