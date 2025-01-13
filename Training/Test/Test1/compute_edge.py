import torch
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
edge_index = torch.tensor(combinations, dtype=torch.long)