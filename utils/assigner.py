import torch
import numpy as np



class GroupAssigner(object):
    
    def __init__(self, criterion=None, n_groups=2, prob=None):
        self.criterion = criterion # func(data) -> group assignment
        self.n_groups = n_groups
        if self.criterion is None:
            self.prob = prob if prob else torch.ones(n_groups).float()/n_groups

    def __call__(self, data):
        if self.criterion is not None:
            group_id = self.criterion(data)
        else:
            group_id = torch.tensor(
                np.random.choice(range(self.n_groups), 1, p=self.prob)
                ).long()
        data.group_id = group_id
        return data

