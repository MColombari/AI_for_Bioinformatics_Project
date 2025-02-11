# First divide it in classes
import random
L = range(2000)
amount = 300
os = [random.choice(L) for _ in range(amount)]
print(os)
os.sort()

n = len(os)
num_classes = 3

split_values = []
for c in range(1, num_classes + 1):
    if c == num_classes:
        split_values.append(os[len(os) - 1])
    else:
        index = (n // num_classes) * c
        split_values.append(os[index - 1])
print(split_values)

list_data_split = []
for c in range(num_classes):
    list_data_split.append([])
    for d in list_of_Data:
        if  (c == 0 and int(d.y) <= split_values[c]) or \
            (c > 0 and int(d.y) <= split_values[c] and int(d.y) > split_values[c-1]):
            d.y = torch.tensor(c)
            list_data_split[c].append(d)

# Now split in train and test.
train_list = []
test_list = []

if percentage_test > 0:
    test_interval = np.floor(1 / percentage_test)
else:
    test_interval = len(list_of_Data) + 1 # we'll never reach it.
# print(test_interval)

for class_list in list_data_split:
    count = 1
    for d in class_list:
        if count >= test_interval:
            test_list.append(d)
            count = 0
        else:
            train_list.append(d)
        count += 1