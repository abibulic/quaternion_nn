import copy
import numpy as np

import torch
from torch.utils.data import Dataset

from graph_network import utils_torch

class QuaternionData(Dataset):

    def __init__(self, paths, args, mean, std):
        self.paths = paths
        self.args = args
        self.mean = mean
        self.std = std
        
    def __getitem__(self, index):
        graph = np.load(self.paths[index], allow_pickle=True).item()
        target = graph['globals'][0][:4]
        #target = (target-self.mean)/self.std
        target = target[np.newaxis, :]
        graph['globals'] = np.zeros(target.shape)

        return graph, target

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def collate_fn(batch):
        data = []
        target = []
        for item in batch:
            data.append(item[0])
            target.append(item[1])

        data = utils_torch.data_dicts_to_graphs_tuple(data)
        target = torch.FloatTensor(np.concatenate(target, axis=0))
        return data, target