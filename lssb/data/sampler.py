import numpy as np
import torch
from torch.utils.data import Sampler
from collections import defaultdict
from tqdm import tqdm

'''
adopted from 

https://github.com/mileyan/simple_shot/blob/5d38fc83e698f11fea56bdfa3e1a8fdde9935e1a/src/datasets/sampler.py
'''

class CategoriesSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n_shot, n_query):

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        
        if isinstance(label, torch.Tensor):
            if label.device.type == 'cuda':
                label = label.cpu()
            label = [x.item() for x in label]
        
        label = np.array(label)
        self.m_ind = []
        unique = np.unique(label)
        unique = np.sort(unique)
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            for c in classes:
                l = self.m_ind[c.item()]
                pos = torch.randperm(l.size()[0])
                batch_gallery.append(l[pos[:self.n_shot]])
                batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]])
            batch = torch.cat(batch_gallery + batch_query)
            yield iter(batch.tolist())
    
