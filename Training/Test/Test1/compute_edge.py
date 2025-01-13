import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from matplotlib import pyplot as plt

# list_list[[], []]
in_list = [1,2,3,42,2,3,4,3,3,45,666,4,4,33]
t = 5
x = list(zip(range(len(in_list)), in_list))

x = sorted(x, key=lambda x: x[1])
combinations = [[],[]]
print(x)

for i in range(len(x)):
    for j in range(i+1, len(x)):
        if(x[j][1] < x[i][1] + 3):
            combinations[0].append(x[i][0])
            combinations[0].append(x[j][0])
            combinations[1].append(x[j][0])
            combinations[1].append(x[i][0])
        else:
            break
print(combinations[0])
print(combinations[1])

print(len(combinations[1]))
print(len(x))

edge_index = torch.tensor(combinations, dtype=torch.long)
x = torch.tensor([list(a) for a in x], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)


def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()
 
 

G = to_networkx(data, to_undirected=True)
visualize_graph(G, color=data.y)