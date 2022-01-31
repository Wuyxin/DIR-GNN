""" This is directly adapted from https://github.com/kohpangwei/group_DRO and adapted"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LossComputer:
    def __init__(self, criterion, is_robust, n_groups, alpha=None, gamma=0.1, adj=None,
                 min_var_weight=0, step_size=0.01, normalize_loss=False, btl=False, device='cpu'):

        self.criterion = criterion
        self.is_robust = is_robust
        self.step_size = step_size

        self.btl = btl # set to false
        self.device = device

        self.n_groups = n_groups 
        # self.group_counts = torch.tensor(dataset.group_counts).to(device)
        # self.group_frac = self.group_counts/self.group_counts.sum()
        self.group_frac = torch.ones(self.n_groups).to(device)/self.n_groups

        # quantities mintained throughout training
        self.adv_probs = torch.ones(self.n_groups).to(device)/self.n_groups

        # The following 4 variables are not used
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.normalize_loss = normalize_loss

    def loss(self, yhat, y, weight=None, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        if weight is not None:
            per_sample_losses = self.criterion(yhat, y, weight, average_across_batch=False)
        else:
            per_sample_losses = self.criterion(yhat, y)

        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count) # this one is actually used
        elif self.is_robust and self.btl:
             actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss

        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss * self.adv_probs # element-wise out
        return robust_loss, self.adv_probs

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0) # avoid nans
        group_loss = (group_map.float() @ losses.view(-1).float())/group_denom.float()
        return group_loss, group_count

