# Standard library imports
import os
import math
import json
import logging
from pathlib import Path
from typing import Union, Sequence, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Local imports
from src.datasets.base import BaseDataset
from src.utils import Array, set_rngs

# Constants
FIXED_SEED = 0
MAX_DATA_ALLOWED = 19020


def create_path_if_not_exists(directory_path):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


class BaseUCI(BaseDataset):
    """
    BaseUCI class for handling UCI datasets.

    Args:
        data_dir (Union[str, Path]): Directory where the data is stored.
        train (bool): Whether to load the training set.
        test_label_counts (dict): Counts of labels in the test set.
        seed (int): Random seed for reproducibility.
        verbose (bool): Verbosity flag.
        partially_observed (bool): Whether the data is partially observed.
        len_historical_dataset (int): Length of the historical dataset.
        llm_cfg: Configuration object for LLM settings.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        train: bool = True,
        test_label_counts: dict = None,
        seed: int = None,
        verbose: bool = False,
        partially_observed: bool = False,
        len_historical_dataset: int = 0,
        llm_cfg=None,
    ) -> None:
        super().__init__()

        # Initialize basic attributes
        self.seed = seed
        self.data_dir = llm_cfg.data_dir
        self.llm_dir = llm_cfg.llm_dir
        self.feature_selected_dir = "feature_selected"
        self.len_historical_dataset = len_historical_dataset
        self.this_dataset_name = self.get_dataset_name()
        self.is_train = train
        self.test_label_counts = test_label_counts
        self.special_case = False
        self.partially_observed = partially_observed
        self.use_mask = False
        self.syn_class = None

        # LLM configuration
        if llm_cfg is not None:
            self.len_pool = llm_cfg.len_pool
            self.llm_name = llm_cfg.llm_name
            self.add_name = llm_cfg.add_name
            self.set_feat_percentage = llm_cfg.set_feat_percentage
            if llm_cfg.pool_set_n_samples is not None:
                self.hist_set_n_samples = min(
                    llm_cfg.pool_set_n_samples, len_historical_dataset
                )
            self.pool_set_type = llm_cfg.pool_set_type
            self.missing_with_relevance = llm_cfg.missing_with_relevance

            if not (
                ("All" == llm_cfg.pool_set_type and self.partially_observed)
                or ("Hist" == llm_cfg.pool_set_type and self.partially_observed)
            ):
                self.use_mask = True

            self.MAX_STEPS = llm_cfg.max_steps
            self.LR_RATE = llm_cfg.lr_rate
            self.CFG_R = llm_cfg.cfg_r
            self.CFG_LORA_ALPHA = llm_cfg.cfg_lora_alpha
            self.PER_DEVICE_BATCH = llm_cfg.per_device_batch
            self.syn_class = llm_cfg.syn_class

        print(self.use_mask)

        # Load and preprocess data
        data_x, data_y, n_test, self.index_original_pd = self.get_original_data()
        self.cols_original = self.get_col_names_original()
        self.n_total_data = len(data_x)
        data_x = data_x[self.get_columns_original_ordered_used()]

        # Handle categorical data
        categorical_data = self.col_cat_original_used
        if len(categorical_data) > 0:
            data_x = pd.get_dummies(
                data_x, columns=categorical_data, prefix=categorical_data
            ).astype(float)
            self.col_dummies = data_x.columns
        else:
            self.col_dummies = None

        self.columns_ordered_used = self.get_columns_ordered_used()
        data_x = data_x[self.columns_ordered_used]

        # Convert to numpy arrays
        self.data = data_x.to_numpy()
        self.targets = data_y.to_numpy()

        # Determine data indices
        if not ("Hist" == llm_cfg.pool_set_type):
            self.final_idx_data_used = (
                - n_test if not self.use_mask else - n_test - self.len_historical_dataset
            )
        else:
            self.final_idx_data_used = - n_test - self.len_historical_dataset

        self.len_data_used = len(self.data[: self.final_idx_data_used])

        # Normalize the inputs
        self.data = self.data.astype(np.float32)
        self.mean = np.mean(self.data[:-n_test], axis=0, keepdims=True)
        self.std = np.std(self.data[:-n_test], axis=0, keepdims=True)
        self.data = (self.data - self.mean) / (self.std + 1e-8)

        # Map labels to start at zero
        label_map = {_class: i for i, _class in enumerate(np.unique(self.targets))}
        label_map = np.vectorize(label_map.get)
        self.targets = label_map(self.targets)

        if verbose:
            self.log_class_frequencies(self.targets, n_test)

        # Split into training and test sets
        if train:
            self.data = self.data[: self.len_data_used]
            self.targets = self.targets[: self.len_data_used]
        else:
            is_test = np.full(len(self.data), False)
            is_test[-n_test:] = True

            rng = np.random.default_rng(seed=seed)
            test_inds = []

            for label, count in test_label_counts.items():
                _test_inds = np.flatnonzero(is_test & (self.targets == label))
                _test_inds = rng.choice(_test_inds, size=count, replace=False)
                test_inds.append(_test_inds)

            test_inds = np.concatenate(test_inds)
            test_inds = rng.permutation(test_inds)

            self.data = self.data[test_inds]
            self.targets = self.targets[test_inds]

        # Create masks for training data
        if train:
            set_rngs(FIXED_SEED)

            def create_random_dataframe(this_columns, n_rows, per_prob):
                data = {
                    column: np.random.choice(
                        [True, False], size=n_rows, p=[per_prob, 1 - per_prob]
                    )
                    for column in this_columns
                }
                for row in range(n_rows):
                    true_count = sum(data[column][row] for column in this_columns)
                    while true_count < 3:
                        random_column = np.random.choice(this_columns)
                        data[random_column][row] = True
                        true_count = sum(data[column][row] for column in this_columns)
                df = pd.DataFrame(data)
                return df

            if not self.missing_with_relevance:
                self.mask_table = create_random_dataframe(
                    self.get_columns_original_ordered_used(),
                    self.n_total_data,
                    self.set_feat_percentage,
                )
            else:
                def create_random_dataframe2(this_columns, n_rows, per_prob):
                    min_col = 3
                    p_min = 0.1
                    p_max = per_prob * 2 - p_min
                    n_cols_unobs = len(this_columns) - min_col
                    all_probs = np.concatenate(
                        [np.ones(min_col,), np.linspace(p_max, p_min, n_cols_unobs)]
                    )
                    data = {
                        this_columns[this_idx]: np.random.choice(
                            [True, False],
                            size=n_rows,
                            p=[all_probs[this_idx], 1 - all_probs[this_idx]],
                        )
                        for this_idx in range(len(this_columns))
                    }
                    df = pd.DataFrame(data)
                    return df

                self.mask_table = create_random_dataframe2(
                    self.get_columns_original_ordered_used(),
                    self.n_total_data,
                    self.set_feat_percentage,
                )

            self.mask_table.index = self.index_original_pd
            set_rngs(seed)

            # Extend mask table if using mask
            if self.use_mask:
                self.mask_table_extended = pd.DataFrame(
                    columns=self.columns_ordered_used
                )
                for col_name_used in self.columns_ordered_used:
                    for col_name_original in self.get_columns_original_ordered_used():
                        if (
                            (col_name_original in self.col_cat_original_used and f"{col_name_original}_" in col_name_used)
                            or col_name_original == col_name_used
                        ):
                            self.mask_table_extended[col_name_used] = self.mask_table[
                                col_name_original
                            ]
            else:
                self.mask_table_extended = self.mask_table.copy()

    def obtain_data_from_columns(self, orig_columns_used, orig_categorical_columns_used):
        """
        Obtain data from specified columns.

        Args:
            orig_columns_used (list): Original columns to use.
            orig_categorical_columns_used (list): Original categorical columns to use.

        Returns:
            data (np.ndarray): Data array.
            targets (np.ndarray): Target array.
        """
        data_x, data_y, n_test, _ = self.get_original_data()
        data_x = data_x[orig_columns_used]
        if len(orig_categorical_columns_used) > 0:
            data_x = pd.get_dummies(
                data_x, columns=orig_categorical_columns_used, prefix=orig_categorical_columns_used
            ).astype(float)

        data = data_x.to_numpy()
        targets = data_y.to_numpy()
        final_idx_data_used = -n_test if not self.use_mask else -n_test - self.len_historical_dataset
        len_data_used = len(data[:final_idx_data_used])

        # Normalize the inputs
        data = data.astype(np.float32)
        mean = np.mean(data[:-n_test], axis=0, keepdims=True)
        std = np.std(data[:-n_test], axis=0, keepdims=True)
        data = (data - mean) / (std + 1e-8)

        # Map labels to start at zero
        label_map = {_class: i for i, _class in enumerate(np.unique(targets))}
        label_map = np.vectorize(label_map.get)
        targets = label_map(targets)

        if self.is_train:
            data = data[:len_data_used]
            targets = targets[:len_data_used]
        else:
            is_test = np.full(len(data), False)
            is_test[-n_test:] = True
            rng = np.random.default_rng(seed=self.seed)
            test_inds = []
            for label, count in self.test_label_counts.items():
                _test_inds = np.flatnonzero(is_test & (targets == label))
                _test_inds = rng.choice(_test_inds, size=count, replace=False)
                test_inds.append(_test_inds)
            test_inds = np.concatenate(test_inds)
            test_inds = rng.permutation(test_inds)
            data = data[test_inds]
            targets = targets[test_inds]
        return data, targets

    def get_original_index_pool_data(self):
        """Get original index of pool data."""
        return self.index_original_pd[:self.len_data_used]

    def get_original_mask_pool_data(self):
        """Get original mask of pool data."""
        return self.mask_table[:self.len_data_used], self.mask_table_extended[:self.len_data_used]

    def get_original_all_pool_data(self):
        """Get all original pool data."""
        data_x, _, n_test, index_original_pd = self.get_original_data()
        data_x = data_x[self.get_columns_original_ordered_used()]
        if self.pool_set_type == 'All':
            new_mask = self.mask_table.copy()
            new_mask[self.col_observed_original_used] = True
            new_mask[self.col_unobserved_original_used] = False
            return data_x[:-n_test], index_original_pd[:-n_test], new_mask[:-n_test]
        elif self.pool_set_type == 'Hist':
            new_mask = self.mask_table.copy()
            new_mask[self.col_observed_original_used] = True
            new_mask[self.col_unobserved_original_used] = False
            return data_x[:self.len_data_used], index_original_pd[:self.len_data_used], new_mask[:self.len_data_used]
        else:
            if self.len_pool == 0:
                return data_x[:self.len_data_used], index_original_pd[:self.len_data_used], self.mask_table[:self.len_data_used]
            else:
                return data_x[:self.len_pool], index_original_pd[:self.len_pool], self.mask_table[:self.len_pool]

    def get_original_historical_data(self):
        """Get original historical data."""
        data_x, _, n_test, index_original_pd = self.get_original_data()
        data_x = data_x[self.get_columns_original_ordered_used()]
        new_mask = self.mask_table.copy()
        new_mask[self.get_columns_original_ordered_used()] = True
        if self.pool_set_type == 'All':
            return data_x[:-n_test], index_original_pd[:-n_test], new_mask[:-n_test]
        else:
            return (
                data_x[self.len_data_used : self.len_data_used + self.hist_set_n_samples],
                index_original_pd[self.len_data_used : self.len_data_used + self.hist_set_n_samples],
                new_mask[self.len_data_used : self.len_data_used + self.hist_set_n_samples],
            )

    def load_cost_data(self, include_y_cost=False, different_x_cost=False, stochastic_cost=False):
        """Load cost data."""
        set_rngs(FIXED_SEED)
        if not self.use_mask:
            feat_used = self.col_unobserved_original_used
        else:
            feat_used = self.get_columns_original_ordered_used()
        if not different_x_cost:
            cost_data = np.ones((self.n_total_data, len(feat_used)))
        else:
            cost_data = np.arange(len(feat_used)).reshape(1, -1).repeat(self.n_total_data, axis=0)
        cost_y = np.zeros((self.n_total_data, 1)) if not include_y_cost else cost_data.sum(1).reshape(-1, 1)
        cost_data = np.concatenate([cost_data, cost_y], axis=1)
        cost_data = cost_data / cost_data.sum(axis=1, keepdims=True)

        if stochastic_cost:
            cost_noise = np.random.rand(self.n_total_data, len(feat_used) + 1) - 0.5
            cost_data = cost_data + cost_noise * cost_data
        set_rngs(self.seed)
        return cost_data

    def get_dataset_name(self):
        """Get dataset name."""
        return

    def get_dataset_path(self):
        """Get dataset path."""
        if 'dim' in self.syn_class and 'corr' in self.syn_class:
            data_path = f'{self.data_dir}/{self.get_dataset_name()}-{self.syn_class}'
        else:
            data_path = f'{self.data_dir}/{self.get_dataset_name()}'
        create_path_if_not_exists(data_path)
        return f'{data_path}/data.csv'

    def get_name_data_llm(self):
        """Get name of LLM data."""
        if not self.missing_with_relevance:
            return f"feat_per_{self.set_feat_percentage}_pool_n_samples_{self.hist_set_n_samples}_pool_type_{self.pool_set_type}{self.add_name}"
        else:
            return f"feat_per_{self.set_feat_percentage}_pool_n_samples_{self.hist_set_n_samples}_pool_type_{self.pool_set_type}_MissR{self.add_name}"

    def get_name_cfg_llm(self):
        """Get LLM configuration name."""
        return f"MAX_STEPS_{self.MAX_STEPS}_LR_{self.LR_RATE}_R_{self.CFG_R}_ALPHA_{self.CFG_LORA_ALPHA}_PB_{self.PER_DEVICE_BATCH}"

    def get_dir_data_llm(self):
        """Get directory for LLM data."""
        return f"{self.llm_dir}/{self.get_dataset_name()}/{self.llm_name}/{self.get_name_data_llm()}/{self.get_name_cfg_llm()}"

    def get_dir_llm_vanilla(self):
        """Get directory for vanilla LLM."""
        return f"{self.get_dataset_name()}/{self.llm_name}/{self.get_name_data_llm()}/{self.get_name_cfg_llm()}"

    def get_dataset_historical_path_jsonl(self):
        """Get historical dataset path in JSONL format."""
        data_path = f'{self.data_dir}/{self.get_dataset_name()}'
        create_path_if_not_exists(data_path)
        data_train = f'{data_path}/data_train_{self.get_name_data_llm()}.jsonl'
        data_val = f'{data_path}/data_val_{self.get_name_data_llm()}.jsonl'
        return data_train, data_val

    def get_project_name(self):
        """Get project name."""
        return f"{self.llm_name}-{self.get_name_data_llm()}-{self.get_name_cfg_llm()}"

    def get_dir_llm(self):
        """Get directory for LLM."""
        dir_used = f'{self.llm_dir}/{self.get_dataset_name()}/{self.llm_name}'
        create_path_if_not_exists(dir_used)
        return dir_used

    def get_dir_llm_checkpoint(self):
        """Get directory for LLM checkpoints."""
        dir_used = f'{self.get_dir_data_llm()}/checkpoint'
        create_path_if_not_exists(dir_used)
        return dir_used

    def get_dir_llm_checkpoint_vllm(self):
        """Get directory for vLLM checkpoints."""
        dir_used = f'{self.get_dir_data_llm()}/vllm_checkpoint'
        create_path_if_not_exists(dir_used)
        return dir_used

    def get_dir_llm_samples(self):
        """Get directory for LLM samples."""
        dir_used = f'{self.get_dir_data_llm()}/samples'
        create_path_if_not_exists(dir_used)
        return dir_used

    def get_dir_llm_outputs(self):
        """Get directory for LLM outputs."""
        dir_used = f'{self.get_dir_data_llm()}/outputs'
        create_path_if_not_exists(dir_used)
        return dir_used

    def get_dir_feature_selected(self):
        """Get directory for selected features."""
        dir_used = f'{self.feature_selected_dir}/{self.get_dataset_name()}'
        create_path_if_not_exists(dir_used)
        return f'{dir_used}/feature_selected_per_{self.set_feat_percentage}.json'

    def get_columns_ordered_used(self):
        """Get ordered list of used columns."""
        if not self.use_mask:
            obs_features, unobs_features = self.get_feature_obsunobs_dummies()
            return obs_features + unobs_features
        else:
            return self.get_features_dummies()

    def get_columns_original_ordered_used(self):
        """Get ordered list of original columns used."""
        if not self.use_mask:
            return list(self.col_observed_original_used + self.col_unobserved_original_used)
        else:
            return list(self.col_observed_original_used)

    def get_startswith_dummies(self, features_list):
        """Get dummy columns starting with specified features."""
        new_list = [
            f"{col_name}_" if col_name in self.col_cat_original_used else col_name
            for col_name in features_list
        ]
        return self.col_dummies[self.col_dummies.str.startswith(tuple(new_list))]

    def get_feature_obsunobs_dummies(self):
        """Get observed and unobserved features with dummies."""
        obs_features = self.col_observed_original_used
        unobs_features = self.col_unobserved_original_used
        if self.col_dummies is not None:
            obs_features_dummies = self.get_startswith_dummies(obs_features)
            unobs_features_dummies_new = self.get_startswith_dummies(unobs_features)
            return list(obs_features_dummies), list(unobs_features_dummies_new)
        else:
            return obs_features, unobs_features

    def get_features_dummies(self):
        """Get features with dummies."""
        if self.col_dummies is not None:
            obs_features_dummies = self.get_startswith_dummies(self.col_observed_original_used)
            return list(obs_features_dummies)
        else:
            return list(self.col_observed_original_used)

    def get_features_indexes_matrix(self):
        """Get feature indexes matrix."""
        if not self.use_mask:
            unobs_features_orig = self.col_unobserved_original_used
            _, features_dummies_used = self.get_feature_obsunobs_dummies()
        else:
            unobs_features_orig, features_dummies_used = self.col_observed_original_used, self.get_features_dummies()

        feature_indexes_unobs = torch.zeros(len(unobs_features_orig), len(features_dummies_used)).float()
        columns_dummies = pd.DataFrame([], columns=features_dummies_used).columns
        for idx, unobs_name in enumerate(unobs_features_orig):
            filter_unobs = columns_dummies.str.startswith(
                f"{unobs_name}_" if unobs_name in self.col_cat_original_used else unobs_name
            )
            feature_indexes_unobs[idx, filter_unobs] = 1.0
        return feature_indexes_unobs

    def get_feature_observed_num(self, feat_importance, original_col_names):
        """Get numerical features observed."""
        idx_feat_importance = math.ceil(len(feat_importance) * self.set_feat_percentage) - 1
        val_feat_importance = feat_importance[feat_importance.argsort()][idx_feat_importance]
        feat_observability = feat_importance <= val_feat_importance
        final_list = list(np.array(original_col_names)[feat_observability])
        return final_list

    def get_features_ordered(self, columns_dummies, feat_importance_dummies, columns_original, th_finish, cat_constraint=True):
        """Get ordered features."""
        columns_original_used = []
        argidx = feat_importance_dummies.argsort()
        for idx_used in argidx:
            current_col = columns_dummies[idx_used]
            for col in columns_original:
                if col in current_col and col not in columns_original_used:
                    columns_original_used.append(col)
                    if cat_constraint:
                        list_finish_constrain = columns_dummies[
                            columns_dummies.str.startswith(tuple(columns_original_used))
                        ]
                    else:
                        list_finish_constrain = columns_original_used
                    break
            if len(list_finish_constrain) >= th_finish:
                break
        return columns_original_used

    def get_feature_observed_num_cat(self, feat_importance, col_names_dummies, original_col_names):
        """Get observed numerical and categorical features."""
        cat_col_original = self.get_col_categorical()
        num_col_original = [col for col in original_col_names if col not in cat_col_original]

        num_col_filter = col_names_dummies.str.startswith(tuple(num_col_original))
        cat_col_filter = col_names_dummies.str.startswith(tuple(cat_col_original))

        cat_col = col_names_dummies[cat_col_filter]
        feat_importance_cat = feat_importance[cat_col_filter]
        num_col = col_names_dummies[num_col_filter]
        feat_importance_num = feat_importance[num_col_filter]

        len_to_reach_cat = math.ceil(len(cat_col) * self.set_feat_percentage)

        # Categorical selection
        orig_cat_col_used = self.get_features_ordered(
            cat_col, feat_importance_cat, cat_col_original, len_to_reach_cat, cat_constraint=True
        )

        orig_num_col_used = self.get_feature_observed_num(feat_importance_num, num_col)

        return orig_num_col_used, orig_cat_col_used[:-1] if len(orig_cat_col_used) >= 1 else []

    def obtain_feature_observed(self, data, target, original_col_names):
        """Obtain observed features based on feature importance."""
        dir_feature_observed = self.get_dir_feature_selected()
        if os.path.exists(dir_feature_observed):
            with open(dir_feature_observed, 'r') as json_file:
                this_json_dict = json.load(json_file)
        else:
            if 'syndata' not in self.this_dataset_name or 'ablation' in self.this_dataset_name:
                categorical_col = self.get_col_categorical()
                data = pd.get_dummies(data, columns=categorical_col, prefix=categorical_col).astype(float)
                columns_extended = data.columns
                data = data.to_numpy()
                mean, std = np.mean(data, axis=0, keepdims=True), np.std(data, axis=0, keepdims=True)
                data = (data - mean) / std
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(data, target)
                feat_importance = clf.feature_importances_

                original_columns = self.get_col_names_original()
                feat_obs_orig_ordered = self.get_features_ordered(
                    columns_extended, feat_importance, original_columns, len(original_columns), cat_constraint=False
                )

                if not self.special_case:
                    if categorical_col is None:
                        num_feat_obs_aux, cat_feat_obs_aux = self.get_feature_observed_num(
                            feat_importance, original_col_names
                        ), []
                    else:
                        num_feat_obs_aux, cat_feat_obs_aux = self.get_feature_observed_num_cat(
                            feat_importance, columns_extended, original_col_names
                        )
                else:
                    col_selected = feat_obs_orig_ordered[
                        : math.floor(len(feat_obs_orig_ordered) * self.set_feat_percentage)
                    ]
                    num_feat_obs_aux = [
                        col for col in feat_obs_orig_ordered if col not in categorical_col and col in col_selected
                    ]
                    cat_feat_obs_aux = [
                        col for col in feat_obs_orig_ordered if col in categorical_col and col in col_selected
                    ]
            else:
                if 'ablation' not in self.this_dataset_name:
                    feat_obs_orig_ordered = ['X1', 'Group', 'X2']
                    cat_feat_obs_aux = []
                    num_feat_obs_aux = ['X1']
                    categorical_col = self.get_col_categorical()
                else:
                    feat_obs_orig_ordered = ['X1', 'Group', 'X2']
                    cat_feat_obs_aux = []
                    num_feat_obs_aux = ['X1']
                    categorical_col = self.get_col_categorical()

            feat_obs_aux = num_feat_obs_aux + cat_feat_obs_aux

            categorical_col = categorical_col if categorical_col is not None else []
            num_feat_obs_aux = [col for col in feat_obs_orig_ordered if col in num_feat_obs_aux]
            cat_feat_obs_aux = [col for col in feat_obs_orig_ordered if col in cat_feat_obs_aux]
            num_feat_unobs_aux = [
                col for col in feat_obs_orig_ordered if col not in num_feat_obs_aux and col not in categorical_col
            ]
            cat_feat_unobs_aux = [
                col for col in feat_obs_orig_ordered if col not in cat_feat_obs_aux and col in categorical_col
            ]
            feat_obs = [col for col in feat_obs_orig_ordered if col in feat_obs_aux]
            feat_unobs = [col for col in feat_obs_orig_ordered if col not in feat_obs_aux]

            this_json_dict = {
                'feature_observed_aux': feat_obs,
                'feature_unobserved_aux': feat_unobs,
                'num_feature_observed_aux': num_feat_obs_aux,
                'cat_feature_observed_aux': cat_feat_obs_aux,
                'num_feature_unobserved_aux': num_feat_unobs_aux,
                'cat_feature_unobserved_aux': cat_feat_unobs_aux,
                'features_ordered': feat_obs_orig_ordered,
                'features_cat_ordered': cat_feat_obs_aux + cat_feat_unobs_aux,
            }

            with open(dir_feature_observed, 'w') as json_file:
                json.dump(this_json_dict, json_file)

        if not self.use_mask:
            features_observed = this_json_dict['feature_observed_aux']
            features_unobserved = this_json_dict['feature_unobserved_aux']
            self.col_cat_original_used = this_json_dict['features_cat_ordered']
        else:
            features_observed = this_json_dict['features_ordered']
            features_unobserved = []
            self.col_cat_original_used = this_json_dict['features_cat_ordered']

        return features_observed, features_unobserved

    def update_observed_data(self, data_x, data_y):
        """Update observed data features."""
        self.col_observed_original_used, self.col_unobserved_original_used = self.obtain_feature_observed(
            data_x, data_y, self.col_observed_original_used
        )

    def get_original_data(self):
        """Get original data. To be implemented in subclass."""
        pass

    def update_columns_data(self, data, get_original, filter_unobs):
        """Update columns of data."""
        data_new = data[self.get_columns_original_ordered_used()]
        if not get_original:
            if self.col_dummies is not None:
                categorical_data = self.col_cat_original_used
                data_new = pd.get_dummies(
                    data_new, columns=categorical_data, prefix=categorical_data
                ).astype(float)
                data_new = data_new.reindex(columns=self.col_dummies, fill_value=0)

            obs_features_dummies, unobs_features_dummies = self.get_feature_obsunobs_dummies()
            if not self.use_mask:
                if not filter_unobs:
                    return data_new[obs_features_dummies + unobs_features_dummies]
                else:
                    return data_new[unobs_features_dummies]
            else:
                return data_new[obs_features_dummies]
        else:
            if not filter_unobs:
                return data_new
            else:
                return data_new[self.col_unobserved_original_used]


    def log_class_frequencies(self, labels: Sequence[int], n_test: int) -> None:
        """
        Report the class frequencies before and after making the train-test split.
        """
        free = np.full(len(labels), True)

        free_train = np.copy(free)
        free_train[-n_test:] = False

        free_test = np.copy(free)
        free_test[:-n_test] = False

        freqs_all = np.array([np.count_nonzero(labels == i) for i in np.unique(labels)])
        freqs_train = [np.count_nonzero(labels[free_train] == i) for i in np.unique(labels)]
        freqs_train = np.array(freqs_train)
        freqs_test = np.array([np.count_nonzero(labels[free_test] == i) for i in np.unique(labels)])

        rel_freqs_all = np.round(freqs_all / np.sum(freqs_all), 2)
        rel_freqs_train = np.round(freqs_train / np.sum(freqs_train), 2)
        rel_freqs_test = np.round(freqs_test / np.sum(freqs_test), 2)

        logging.info("Before split: " + str(freqs_all) + " " + str(rel_freqs_all))
        logging.info("Train after split: " + str(freqs_train) + " " + str(rel_freqs_train))
        logging.info("Test after split: " + str(freqs_test) + " " + str(rel_freqs_test))


    def process_data(self, data_pd):
        self.col_observed_original_used    = self.get_col_names_original()
        self.col_unobserved_original_used  = None

        data_pd = data_pd.sample(frac=1, random_state=0)[:MAX_DATA_ALLOWED]
        index_original_pd = data_pd.index

        categorical_columns = self.get_col_categorical()
        numerical_columns   = self.get_col_names_original()
        if categorical_columns is not None:
            numerical_columns = [name for name in self.col_observed_original_used if name not in categorical_columns]

        data_pd[numerical_columns] = data_pd[numerical_columns].round(5)

        data_y = data_pd['y']
        data_x = data_pd.drop('y', axis=1)

        self.update_observed_data(data_x, data_y)
        n_test = int(0.3 * len(data_x))
        return data_x, data_y, n_test, index_original_pd 

class Magic(BaseUCI):
    def get_dataset_name(self):
        return 'magic'
    
    def get_col_names_original(self):
        return ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
    
    def get_col_categorical(self):
        return None
    
    def get_original_data(self) -> Tuple[np.ndarray, int]:
        """
        Use a fixed 70-30 train-test split.

        References:
            https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
        """
        create_path_if_not_exists(self.data_dir)
        if not os.path.exists(self.get_dataset_path()):
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
            data_pd = pd.read_csv(url, header=None)
            data_pd.columns = self.get_col_names_original() + ['y']
            data_pd.to_csv(self.get_dataset_path(), index=False)
        else:
            data_pd = pd.read_csv(self.get_dataset_path())

        return self.process_data(data_pd)

class Adult(BaseUCI):
    def get_dataset_name(self):
        return 'adult'
    
    def get_col_names_original(self):
        return ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    
    def get_col_categorical(self):
        return ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    
    def get_original_data(self) -> Tuple[np.ndarray, int]:
        create_path_if_not_exists('data')

        if not os.path.exists(self.get_dataset_path()):
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            data_pd = pd.read_csv(url, names = self.get_col_names_original() + ['income'], sep=",\s*", engine='python')
            data_pd['y'] = data_pd['income'].map({'<=50K': 0, '>50K': 1})
            data_pd = data_pd.drop('income', axis=1)
            data_pd.to_csv(self.get_dataset_path(), index=False)
        else:
            data_pd = pd.read_csv(self.get_dataset_path())
        
        return self.process_data(data_pd)

class Housing(BaseUCI):
    def get_dataset_name(self):
        return 'housing'
    
    def get_col_names_original(self):
        return ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    def get_col_categorical(self):
        return None
    
    def get_original_data(self) -> Tuple[np.ndarray, int]:
        create_path_if_not_exists('data')

        if not os.path.exists(self.get_dataset_path()):
            data_pd = datasets.fetch_california_housing(as_frame=True).frame
            data_pd['y'] = (data_pd['MedHouseVal'] >= data_pd['MedHouseVal'].median()).astype(float)
            data_pd = data_pd.drop('MedHouseVal', axis=1)
            data_pd.to_csv(self.get_dataset_path(), index=False)
        else:
            data_pd = pd.read_csv(self.get_dataset_path())

        return self.process_data(data_pd)

class Cardio(BaseUCI):
    def get_dataset_name(self):
        return 'cardio'
    
    def get_col_names_original(self):
        return ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco' , 'active']
    
    def get_col_categorical(self):
        return None
    
    def get_original_data(self) -> Tuple[np.ndarray, int]:
        create_path_if_not_exists('data')
        data_pd = pd.read_csv(self.get_dataset_path(), delimiter=';', nrows = 12000)
        data_pd = data_pd[self.get_col_names_original() + ['cardio']].rename(columns = {'cardio': 'y'})
        return self.process_data(data_pd)

class Banking(BaseUCI):
    def get_dataset_name(self):
        return 'banking'
    
    def get_col_names_original(self):
        #return ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day_of_week',
        #        'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
        return ['job', 'contact', 'marital', 'poutcome', 'campaign', 'day_of_week', 'pdays', 'duration']
    def get_col_categorical(self):
        return ['job', 'contact', 'marital','poutcome']
        #return ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    def get_original_data(self) -> Tuple[np.ndarray, int]:
        create_path_if_not_exists('data')
        self.special_case = True
        if not os.path.exists(self.get_dataset_path()):
            from ucimlrepo import fetch_ucirepo 
            bank_marketing = fetch_ucirepo(id=222)
            data_pd = pd.concat([bank_marketing['data']['features'], bank_marketing['data']['targets']], axis = 1).dropna()
            data_pd.to_csv(self.get_dataset_path(), index=False)
        else:
            data_pd = pd.read_csv(self.get_dataset_path())
        data_pd = data_pd[self.get_col_names_original() + ['y']]
        return self.process_data(data_pd)
    

def create_data(num_samples = 1000, slope = 1, corr = .97, sep = 1.):
    import numpy as np    
    def get_gaussian(rho, mu, size):
        def correlated_gaussian(rho, size = 1, mu= [0, 0], sigma = [1, 1]):
            cov_matrix = np.array([[sigma[0]**2, rho * sigma[0] * sigma[1]],
                                   [rho * sigma[0] * sigma[1], sigma[1]**2]])
            L = np.linalg.cholesky(cov_matrix)
            uncorrelated = np.random.normal(size=(size, 2))
            correlated = np.dot(uncorrelated, L.T) + mu
            return correlated
        
        # Example usage
        mu = np.array([mu, 0])  # Mean
        sigma = np.array([1, 1])  # Standard deviation
        #rho = 0.0 # Correlation coefficient
        return correlated_gaussian(rho, mu = mu, size=size)

    def classify_all_data(all_data, slope):
        def classify_point(x, y, slope):
            line_y = slope * x
            if y > line_y:
                return 1
            else:
                return 0
        return  np.array([classify_point(point[0], point[1], slope) for point in all_data])
    #sep = 1.5
    #corr = .97
    #num_samples = 1000
    # Save current random state
    random_state = np.random.get_state()
    np.random.seed(0)  # You can use any integer value as the seed
    g1 = get_gaussian(corr, sep, num_samples)
    g2 = get_gaussian(corr, -sep, num_samples)
    np.random.set_state(random_state)

    pred_g1 = classify_all_data(g1, slope)    
    pred_g2 = classify_all_data(g2, slope)    

    pd1 = pd.DataFrame(g1, columns = ['X1', 'X2'])
    pd1['Group'] = 0
    pd1['y'] = pred_g1

    pd2 = pd.DataFrame(g2, columns = ['X1', 'X2'])
    pd2['Group'] = 1
    pd2['y'] = pred_g2

    pd_final = pd.concat([pd1, pd2], axis = 0).reset_index(drop=True)
    return pd_final


class SynCorr(BaseUCI):
    def get_dataset_name(self):
        return 'syndatacorr'
    
    def get_col_names_original(self):
        return ['X1', 'X2', 'Group']
                
    def get_col_categorical(self):
        return []

    def get_original_data(self) -> Tuple[np.ndarray, int]:

        create_path_if_not_exists(self.data_dir)
        if not os.path.exists(self.get_dataset_path()):
            if self.syn_class == "low":
                data_pd = create_data(num_samples = 2500, slope = 999999999999999, corr = .97)
            if self.syn_class == "mid":
                data_pd = create_data(num_samples = 2500, slope = 1e-12, corr = .97)
            if self.syn_class == "high":
                data_pd = create_data(num_samples = 2500, slope = 1, corr = .97)

            data_pd.columns = self.get_col_names_original() + ['y']
            #n_test = len(test_data)
        else:
            data_pd = pd.read_csv(self.get_dataset_path())

        return self.process_data(data_pd)

class SynNoCorr(BaseUCI):
    def get_dataset_name(self):
        return 'syndatanocorr'
    
    def get_col_names_original(self):
        return ['X1', 'X2', 'Group']
                
    def get_col_categorical(self):
        return []

    def get_original_data(self) -> Tuple[np.ndarray, int]:

        create_path_if_not_exists(self.data_dir)
        if not os.path.exists(self.get_dataset_path()):
            if self.syn_class == "mid":
                data_pd = create_data(num_samples = 2500, slope = 1e-12, corr = .0)
            data_pd.columns = self.get_col_names_original() + ['y']
            #n_test = len(test_data)
        else:
            data_pd = pd.read_csv(self.get_dataset_path())

        return self.process_data(data_pd)

import re
def parse_string(s):
    # Pattern to extract integer after 'dim' and float after 'corr'
    dim_pattern = r'dim(\d+)'
    corr_pattern = r'corr([\d.]+)'
    
    # Search for patterns in the string
    dim_match = re.search(dim_pattern, s)
    corr_match = re.search(corr_pattern, s)
    
    # Extract and convert the matches to appropriate types
    dim_value = int(dim_match.group(1)) if dim_match else None
    corr_value = float(corr_match.group(1)) if corr_match else None
    
    return dim_value, corr_value
    
from scipy.stats import pearsonr
def generate_multidimensional_data(N, D, correlation_AB, threshold=0.5):
    """
    Generate synthetic data with multidimensional A and B with the specified correlation.
    
    Parameters:
    N (int): Number of samples.
    D (int): Number of dimensions for A and B.
    correlation_AB (float): Desired correlation between A and B.
    threshold (float): Threshold for logistic transformation.
    
    Returns:
    DataFrame: Contains columns A, B, and C.
    """
    # Mean and standard deviation for A and B
    mean_A, std_A = np.zeros(D), np.ones(D)
    mean_B, std_B = np.zeros(D), np.ones(D)

    # Create the covariance matrix for A and B
    # This will have the correlation_AB on the off-diagonals between corresponding dimensions
    cov_A = np.eye(D) * std_A**2
    cov_B = np.eye(D) * std_B**2
    cov_AB = np.eye(D) * correlation_AB * std_A * std_B

    # Combine into a full covariance matrix
    cov = np.block([[cov_A, cov_AB],
                    [cov_AB, cov_B]])

    # Generate multivariate normal data for A and B
    mean = np.concatenate([mean_A, mean_B])
    data = np.random.multivariate_normal(mean, cov, N)
    A = data[:, :D]
    B = data[:, D:]

    # Orthogonalize B with respect to A in a multidimensional sense
    B_orthogonal = B - np.dot(np.dot(A, np.linalg.pinv(A)), B)

    # Generate C based on a non-linear transformation of B_orthogonal
    logits = 1 / (1 + np.exp(-np.sum(B_orthogonal, axis=1)))
    C = (logits > threshold).astype(int)

    # Create a DataFrame
    df = pd.DataFrame({'A': list(A), 'B': list(B), 'C': C})
    
    # Calculate correlations between the sum of dimensions to reduce dimensionality
    corr_AB = pearsonr(A.flatten(), B.flatten())[0]
    corr_BC = pearsonr(np.sum(B, axis=1), C)[0]
    corr_AC = pearsonr(np.sum(A, axis=1), C)[0]
    
    return df, corr_AB, corr_BC, corr_AC

def transform_to_good_df(df):
    # Expand the lists in column A into separate columns
    A_expanded = pd.DataFrame(df['A'].tolist(), index=df.index)
    A_expanded.columns = [f'A{i+1}' for i in range(A_expanded.shape[1])]
    
    # Expand the lists in column B into separate columns
    B_expanded = pd.DataFrame(df['B'].tolist(), index=df.index)
    B_expanded.columns = [f'B{i+1}' for i in range(B_expanded.shape[1])]
    
    # Combine the expanded columns with the original DataFrame
    df_transformed = pd.concat([A_expanded, B_expanded, df['C']], axis=1)
    return df_transformed

class AblationSynData(BaseUCI):
    def get_dataset_name(self):
        return 'ablationsyndata'
    
    def get_col_names_original(self):
        all_cols = []
        for i in range(self.syn_dim_value):
            all_cols += ['A%s' % (i+1)]
        for i in range(self.syn_dim_value):
            all_cols += ['B%s' % (i+1)]
        return all_cols
                
    def get_col_categorical(self):
        return []

    def get_original_data(self) -> Tuple[np.ndarray, int]:
        N = 8000
        #import pdb
        #pdb.set_trace()
        dim_value, corr_value = parse_string(self.syn_class)
        self.syn_dim_value = dim_value
        create_path_if_not_exists(self.data_dir)
        if not os.path.exists(self.get_dataset_path()):
            data_pd, corr_AB, corr_BC, corr_AC = generate_multidimensional_data(N, dim_value, corr_value)
            data_pd = transform_to_good_df(data_pd)
            #import pdb
            #pdb.set_trace()
            data_pd.columns = self.get_col_names_original() + ['y']
            #n_test = len(test_data)
        else:
            data_pd = pd.read_csv(self.get_dataset_path())

        return self.process_data(data_pd)


def undummify2(df, prefix_to_collapse):
    cols2collapse = {}
    prefix_sep = '_'
    for col in df.columns:
        not_prefix = True
        for prefix in prefix_to_collapse:
            if prefix in col:
                cols2collapse[prefix] = True
                not_prefix = False
        if not_prefix:
            cols2collapse[col] = False 
    print(cols2collapse)
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df