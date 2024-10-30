import numpy as np
import pandas as pd
import torch
from src.utils import Array
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import os


class BaseDataset(Dataset):
    """
    Note: we can change self.__class__.__name__ in __init__() to ensure self.raw_folder points to
    the right place (see PyTorch reference below).

    References:
        https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
        https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html
        https://stackoverflow.com/a/21220030
    """
    def __init__(self) -> None:
        self.partially_observed = False
        self.obtain_gen_data    = False
        self.obtain_cost_data   = False
        self.obtain_mask_data   = False
        self.len_observed       = None

    def __getitem__(self, index: int) -> Tuple[Array, Array, Optional[Array]]:
        
        batch = {}
        batch['targets']    = self.targets[index]
        batch['inputs']     = self.data[index] if not self.partially_observed or self.obtain_mask_data \
                                                 else self.data[index][:self.len_observed]
        if self.obtain_gen_data:
            batch['gen_unobs']  = self.gen_data[index]
        if self.obtain_cost_data:
            batch['cost_inputs'] = self.cost_data[index]
        if self.obtain_mask_data:
            batch['mask_inputs']        = self.mask_data_extended[index]
            batch['mask_inputs_normal'] = self.mask_data[index]
        return batch

    def __len__(self) -> int:
        return len(self.data)

    def numpy(self) -> None:
        if isinstance(self.data, Tensor):
            self.data = self.data.numpy()

        if isinstance(self.targets, Tensor):
            self.targets = self.targets.numpy()

    def torch(self) -> None:
        if isinstance(self.data, np.ndarray):
            self.data = torch.tensor(self.data)

        if isinstance(self.targets, np.ndarray):
            self.targets = torch.tensor(self.targets)

    def to(self, device: str) -> None:
        if isinstance(self.data, Tensor):
            self.data = self.data.to(device)

        if isinstance(self.targets, Tensor):
            self.targets = self.targets.to(device)

    def reset_config(self) -> None:
        self.partially_observed = False
        self.obtain_gen_data    = False
        self.obtain_cost_data   = False
        self.obtain_mask_data   = False

    def is_partially_observed(self) -> None:
        self.partially_observed = True
        obs_features, _ = self.get_feature_obsunobs_dummies()
        #self.len_observed = len(self.col_observed_original_used)
        self.len_observed = len(obs_features)

    def is_obtain_gen_data(self) -> None:
        self.obtain_gen_data = True

    def is_obtain_cost_data(self) -> None:
        self.obtain_cost_data = True

    def is_obtain_mask_data(self) -> None:
        self.obtain_mask_data = True

    # def update_gen_data(self, gen_data: Tensor, index_array_gen: np.array) -> None:
    #     self.gen_data = gen_data
    #     self.index_array_gen = index_array_gen

    def update_gen_data(self, gen_data: Tensor) -> None:
        self.gen_data = gen_data

    def update_cost_data(self, cost_data: Tensor) -> None:
        self.cost_data = cost_data

    def update_mask_data(self, mask_data: Tensor, mask_data_extended: Tensor) -> None:
        self.mask_data          = mask_data
        self.mask_data_extended = mask_data_extended 

    def get_unnormalized_data(self, subset_inds: np.array) -> Tuple[Array, Array]:
        return self.data[subset_inds], self.targets[subset_inds]
    
    def modify_data(self, feature_selected, idx_feature_selected: np.array, imputation_type: str = 'LLM', mask_inputs = None):
        #gen_data_used = self.gen_data[self.index_array_gen[idx_feature_selected]].mean(axis = 1)
        # import pdb
        # pdb.set_trace()
        if feature_selected is not None:
            gen_data_used = self.gen_data[idx_feature_selected].mean(axis = 1)
            if not self.use_mask:
                if imputation_type == 'LLM':
                    self.data[idx_feature_selected, self.len_observed:] = \
                        feature_selected * self.data[idx_feature_selected, self.len_observed:] + (1 - feature_selected) * gen_data_used
                elif imputation_type == 'zero':
                    self.data[idx_feature_selected, self.len_observed:] = \
                            feature_selected * self.data[idx_feature_selected, self.len_observed:]
            else:
                mask_used = mask_inputs[idx_feature_selected]
                if imputation_type == 'LLM':
                    self.data[idx_feature_selected] = \
                        mask_used * self.data[idx_feature_selected] + \
                        (1 - mask_used) * (self.data[idx_feature_selected] * feature_selected + gen_data_used * (1 - feature_selected) )  
                elif imputation_type == 'zero':
                    self.data[idx_feature_selected] = \
                        mask_used * self.data[idx_feature_selected] + \
                        (1 - mask_used) * (self.data[idx_feature_selected] * feature_selected + 0  * (1 - feature_selected) ) 
        else:
            mask_used = mask_inputs[idx_feature_selected]
            self.data[idx_feature_selected] = mask_used * self.data[idx_feature_selected]