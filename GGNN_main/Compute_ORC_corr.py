# description: esempio di utilizzo della classe ORC con matrice di adiacenza generata da correlazione

import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import networkx as nx
import pandas as pd
import time
import numpy as np
from GeometricNetworkAnalysis.ORC import ORC

# Nuova versione: genera la matrice di adiacenza dai dati
input_folder = "./data/ov_tcga2/out/"
omics=["RNA","CNA","Methyl"]
correlation_threshold = 0.7  # Soglia di correlazione
for omic in omics:
    input_name = input_folder + omic + ".csv"
    tb = pd.read_csv(input_name, header=0, index_col=0)
    test = tb.index.tolist()

    # Calcola la matrice di correlazione tra i geni
    corr_matrix = tb.T.corr(method='pearson')
    # Costruisci la matrice di adiacenza binaria
    adj_s = (abs(corr_matrix) > correlation_threshold).astype(int)
    np.fill_diagonal(adj_s.values, 0)  # Rimuovi autocollegamenti

    # Costruisci il grafo
    G = nx.from_numpy_array(adj_s.values)
    G = nx.relabel_nodes(G, dict(enumerate(corr_matrix.index)))
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    results = []
    start = time.time()

    for i in range(tb.shape[1]):
        nx.set_node_attributes(G, {k: tb.loc[k, tb.columns[i]] + 1e-6 for k in G}, name='weight')
        orc = ORC(G, verbose="ERROR")
        edge_curvatures = orc.compute_curvature_edges()
        end = time.time()
        results.append([edge_curvatures[key] for key in sorted(edge_curvatures.keys())])
        print(omic, i + 1, "/", tb.shape[1], ",t=", time.strftime("%H:%M:%S", time.gmtime(end - start)), "/", time.strftime("%H:%M:%S", time.gmtime((end - start) / (i + 1) * tb.shape[1])))

    out = pd.DataFrame(results, index=tb.columns, columns=sorted(edge_curvatures.keys()))
    output_name = input_folder + omic + "_curv.csv"
    out.to_csv(output_name)
