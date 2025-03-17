import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    '''
    We refered to the open source code https://github.com/adambielski/siamese-triplet/blob/master/datasets.py#L146-L183 to implement the sampler.
    '''
    def __init__(self, labels, labels_set, batch_size):
        self.labels_set = labels_set
        self.batch_size = batch_size
        
        self.label_to_indices = {label: np.where(np.array(labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.sampling_num = self.batch_size * len(self.labels_set)
        self.dataset_len = len(labels)
        

    def __iter__(self):
        self.count = 0
        while self.count + self.sampling_num <= self.dataset_len:
            indices = []
            for class_ in self.labels_set:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_]+self.batch_size])
                self.used_label_indices_count[class_] += self.batch_size
                if self.used_label_indices_count[class_] + self.batch_size > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.sampling_num


    def __len__(self):
        return self.dataset_len // self.sampling_num
    
    
def get_labels_set(dataset):
    loader = DataLoader(dataset)
    labels = []
    for _, label in loader:
        labels.append(label.item())
    
    labels_set = list(set(labels))
    
    return labels, labels_set