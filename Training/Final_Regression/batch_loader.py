from torch.utils.data import BatchSampler
import random
import math
import numpy as np

class RandomBatchSampler(BatchSampler):
    def __init__(self, seed, dataset, batch_size, threshold):
        random.seed(seed)
        self.dataset = dataset
        self.batch_size = batch_size
        self.threshold = threshold

        self.labels = [int(dataset[i].y) for i in range(len(dataset))]
        self.indices = list(range(len(self.labels)))

    def __iter__(self):
        indices = self.indices.copy()
        random.shuffle(indices)
        batch = []
        for idx in indices:
            label = self.labels[idx]
            if all(abs(label - self.labels[other]) >= self.threshold for other in batch):
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

    def __len__(self):
        return len(self.dataset) // self.batch_size


class DeterministicDistantBatchSampler(BatchSampler):
    # The drowback of this approach is that is we are not sure to use all the data, and the way
    # we chose the batch is deterministic, and don't change.
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.dataset_size = len(self.dataset)
        self.batch_size = batch_size
        self.offset = 0
        self.number_of_batch = math.floor(self.dataset_size / self.batch_size)

        # Order of element based on OS.
        self.orderedIndex = np.argsort(np.array([int(dataset[i].y) for i in range(len(dataset))])).tolist()

    def __iter__(self):
        batch = []
        if self.offset >= self.number_of_batch:
            self.offset = 0
        for i in range(self.batch_size):
            data_index = (self.number_of_batch * i) + self.offset
            batch.append(self.orderedIndex[data_index])
        self.offset += 1
        print(f"{batch}\t{[self.dataset[b].y for b in batch]}")
        yield batch

    def __len__(self):
        return self.number_of_batch
