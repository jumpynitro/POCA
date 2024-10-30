import numpy as np
import pandas as pd
import torch
from numpy.random import Generator
from omegaconf import DictConfig
from src.datasets.base import BaseDataset
from torch.utils.data import DataLoader, Dataset, Subset
from torch import Tensor
from typing import List, Sequence, Tuple, Union
import os
import signal
import concurrent.futures
import multiprocessing as mp
import subprocess
import io

class Data:
    """
    Specifying label_counts:
    - label_counts is a dictionary of dictionaries.
    - label_counts[subset] specifies how many examples we want from each class for the given subset.
    - The simplest way to specify label_counts[subset] is with integer keys.
        - Example: we want 5 examples from class 0 and 10 example from class 1.
        - Solution: d = {0: 5, 1: 10}.
    - If there are lots of classes, it becomes verbose to express the dictionary this way.
    - Instead we can use a special key to specify want we want for multiple classes at once.
        - Example: we want 5 examples from class 0, 10 examples from 2 classes in (1, 3, 4)
          and 15 examples from 3 classes in range(5, 10).
        - Solution: d = {0: 5, "2_classes_in_(1,3,4)": 10, "3_classes_in_range(5,10)": 15}.

    Specifying label_map:
    - label_map is a dictionary specifying how we want to redefine the classes in our dataset.
    - As for label_counts, we can use integer keys but also optionally one special key.
        - Example: we want to map 0 to 1, 1 to 2, and all others in range(10) to 0.
        - Solution: d = {0: 1, 1: 2, "rest_in_range(10)": 0}.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        batch_sizes: Union[dict, DictConfig],
        label_counts: Union[dict, DictConfig],
        len_historical_dataset: int,
        rng: Generator,
        seed: int,
        label_map: Union[dict, DictConfig] = None,
        test_classes_to_remove: Sequence[int] = None,
        loader_kwargs: Union[dict, DictConfig] = None,
        partially_observed: bool = False,
        llm_cfg = None,
    ) -> None:
        self.columns_order = None
        self.main_dataset = dataset(train=True,
                                    partially_observed = partially_observed,
                                    len_historical_dataset = len_historical_dataset,
                                    seed = seed,
                                    llm_cfg = llm_cfg)
        self.test_dataset = dataset(train=False,
                                    partially_observed = partially_observed,
                                    len_historical_dataset = len_historical_dataset,
                                    seed = seed,
                                    llm_cfg = llm_cfg)
        self.dir_llm  = self.main_dataset.get_dir_llm()
        self.dir_llm_checkpoint = self.main_dataset.get_dir_llm_checkpoint()
        self.dir_llm_samples = self.main_dataset.get_dir_llm_samples()
        self.obs_features_names_original, self.unobs_features_names_original =\
              self.main_dataset.col_observed_original_used, self.main_dataset.col_unobserved_original_used
        self.obs_features_names, self.unobs_features_names = self.main_dataset.get_feature_obsunobs_dummies()

        self.input_shape_po = len(self.obs_features_names)
        self.input_shape_total  = len(self.obs_features_names + self.unobs_features_names)

        self.inds = Data.initialize_subset_indices(self.main_dataset.targets, label_counts, rng)

        if label_map != None:
            Data.map_labels(self.main_dataset, label_map)
            Data.map_labels(self.test_dataset, label_map)

        if test_classes_to_remove != None:
            Data.remove_classes(self.test_dataset, test_classes_to_remove)

        self.batch_sizes = batch_sizes
        self.loader_kwargs = loader_kwargs if loader_kwargs != None else {}

    @staticmethod
    def get_input_shape_po(self):
        return self.input_shape_po

    @property
    def input_shape(self) -> Union[Tuple[int], torch.Size]:
        return self.main_dataset.data.shape[1:]

    @property
    def n_classes(self) -> int:
        if isinstance(self.main_dataset.targets, np.ndarray):
            return len(np.unique(self.main_dataset.targets))
        else:
            return len(torch.unique(self.main_dataset.targets))

    @property
    def n_train_labels(self) -> int:
        return len(self.inds["train"])

    @staticmethod
    def initialize_subset_indices(
        labels: np.ndarray, label_counts: Union[dict, DictConfig], rng: Generator
    ) -> dict:
        """
        Create a dictionary, inds, where inds[subset] is a list of indices chosen such that the
        number of labels from each class is as specified in label_counts[subset].
        """
        inds = {}
        free_inds = np.arange(len(labels))

        for subset, subset_label_counts in label_counts.items():
            subset_label_counts = Data.preprocess_label_counts(subset_label_counts, rng)
            inds[subset] = Data.select_inds_to_match_label_counts(
                free_inds, labels[free_inds], subset_label_counts, rng
            )
            free_inds = np.setdiff1d(free_inds, inds[subset])

        return inds

    @staticmethod
    def preprocess_label_counts(label_counts: Union[dict, DictConfig], rng: Generator) -> dict:
        """
        As noted in the class docstring, label_counts can have special, non-integer keys. This
        function converts label_counts into an equivalent dictionary with only integer keys.
        """
        processed_label_counts = {}

        for class_or_class_set, count in label_counts.items():
            if isinstance(class_or_class_set, int):
                _class = class_or_class_set
                counts_to_add = {_class: count}

            elif isinstance(class_or_class_set, str):
                class_set = class_or_class_set
                n_classes, _, _, class_set = class_set.split("_")

                n_classes = int(n_classes)
                class_set = eval(class_set)

                if n_classes < len(class_set):
                    class_set = rng.choice(list(class_set), n_classes, replace=False)

                counts_to_add = {_class: count for _class in sorted(class_set)}

            else:
                raise ValueError

            assert np.all([k not in processed_label_counts.keys() for k in counts_to_add.keys()])

            processed_label_counts = {**processed_label_counts, **counts_to_add}

        return processed_label_counts

    @staticmethod
    def select_inds_to_match_label_counts(
        inds: np.ndarray,
        labels: np.ndarray,
        label_counts: Union[dict, DictConfig],
        rng: Generator,
    ) -> List[int]:
        """
        There's no need to return rng. Its internal state gets updated each time it is used.
        >>> f = lambda rng: rng.integers(0, 10)
        >>> rng = np.random.default_rng(0)
        >>> print(f(rng), f(rng))  # -> (8, 6)
        """
        shuffle_inds = rng.permutation(len(inds))
        inds = inds[shuffle_inds]
        labels = labels[shuffle_inds]

        selected_inds = []

        for label, count in label_counts.items():
            label_inds = inds[np.flatnonzero(labels == label)]
            assert len(label_inds) >= count
            selected_inds.append(label_inds[:count])

        selected_inds = np.concatenate(selected_inds)
        selected_inds = rng.permutation(selected_inds)

        return list(selected_inds)

    @staticmethod
    def map_labels(dataset: Dataset, label_map: Union[dict, DictConfig]) -> None:
        """
        Apply label_map to update dataset.targets. Also keep a copy of the old labels.
        """
        label_map = Data.preprocess_label_map(label_map)
        labels = dataset.targets

        # Copying here ensures the correct behavior if label_map is an empty dictionary.
        mapped_labels = np.copy(labels)

        for old_label, new_label in label_map.items():
            old_label_inds = np.flatnonzero(labels == old_label)
            mapped_labels[old_label_inds] = new_label

        dataset.original_targets = labels
        dataset.targets = mapped_labels

    @staticmethod
    def preprocess_label_map(label_map: Union[dict, DictConfig]) -> Union[dict, DictConfig]:
        """
        As noted in the class docstring, label_map can have a special, non-integer key. This
        function converts label_map into an equivalent dictionary with only integer keys.
        """
        if not isinstance(label_map, (dict, DictConfig)):
            return {}

        special_keys = [key for key in label_map.keys() if isinstance(key, str)]

        assert len(special_keys) <= 1, "Up to one special key allowed in a label map"

        if len(special_keys) == 1:
            special_key = special_keys[0]

            _, _, old_labels = special_key.split("_")
            old_labels = eval(old_labels)

            new_label_map = {old_label: label_map[special_key] for old_label in old_labels}

            for key, new_label in label_map.items():
                if key != special_key:
                    new_label_map[key] = new_label

            label_map = new_label_map

        return label_map

    @staticmethod
    def remove_classes(dataset: Dataset, classes_to_remove: Sequence[int]) -> None:
        inds_to_keep = np.flatnonzero([label not in classes_to_remove for label in dataset.targets])
        dataset.data = dataset.data[inds_to_keep]
        dataset.targets = dataset.targets[inds_to_keep]
        dataset.original_targets = dataset.original_targets[inds_to_keep]

    def numpy(self) -> None:
        self.main_dataset.numpy()
        self.test_dataset.numpy()

    def torch(self) -> None:
        self.main_dataset.torch()
        self.test_dataset.torch()

    def to(self, device: str) -> None:
        self.main_dataset.to(device)
        self.test_dataset.to(device)

    # def update_gen_data(self, gen_data: Tensor, index_gen_data: np.array) -> None:
    #     self.main_dataset.update_gen_data(gen_data, index_gen_data)
    def update_gen_data(self, gen_data: Tensor) -> None:
        self.main_dataset.update_gen_data(gen_data)        

    def update_cost_data(self, cost_data: Tensor) -> None:
        self.main_dataset.update_cost_data(cost_data)

    def update_mask_data(self, mask_table: Tensor, mask_table_extended: Tensor) -> None:
        self.main_dataset.update_mask_data(mask_table, mask_table_extended)
        
    def get_dataset(self, subset: str, partially_observed: bool = False,  obtain_gen_data: bool = False,
                                        obtain_cost_data: bool = False, use_mask: bool = False) -> Tuple[Dataset, np.ndarray]:
        subset_inds = None
        if subset == "test":
            dataset_used = self.test_dataset
        else:
            subset_inds = self.inds[subset]
            dataset_used = self.main_dataset

        dataset_used.reset_config()

        if partially_observed:
            dataset_used.is_partially_observed()
        
        if obtain_gen_data:
            dataset_used.is_obtain_gen_data()

        if obtain_cost_data:
            dataset_used.is_obtain_cost_data()

        if use_mask:
            dataset_used.is_obtain_mask_data()

        dataset = Subset(self.main_dataset, subset_inds) if subset != 'test' else dataset_used
        return dataset, subset_inds

    def get_loader(self, subset: str, partially_observed: bool = False,
                                            obtain_gen_data: bool = False,
                                            obtain_cost_data: bool = False,
                                            use_mask: bool = False) -> DataLoader:
        
        dataset, _ = self.get_dataset(subset, partially_observed, obtain_gen_data, obtain_cost_data, use_mask)

        batch_size = self.batch_sizes[subset]
        batch_size = len(dataset) if batch_size == -1 else batch_size

        shuffle = True if subset in {"train", "target"} else False

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **self.loader_kwargs)
    
    def get_features_indexes_matrix(self):
        return self.main_dataset.get_features_indexes_matrix()
    
    def get_pd_dataset(self, subset: str = None, get_partially_observed = False) -> Tuple[pd.DataFrame, np.array]:
        data, _, _, _ = self.main_dataset.get_original_data()
        if subset is not None:
            data            = data[np.array(self.inds[subset])]
        if get_partially_observed:
            data[self.unobs_features_names_original] = np.nan
        return data
    
    def get_original_index_pool_data(self):
        return self.main_dataset.get_original_index_pool_data()

    def get_index_original_pd(self, subset: str):
        subset_inds = self.inds[subset]
        return self.main_dataset.index_original_pd[subset_inds]
        
    def get_index_gen_array(self, subset: str):
        subset_inds = self.inds[subset]
        len_total  = len(self.main_dataset.index_original_pd)
        index_used = np.zeros((len_total,))
        index_used[subset_inds] = np.arange(len(subset_inds))
        return index_used.astype(int)

    def normalize_data(self, gen_data: np.array, partially_observed: str = False) -> np.array:
        if hasattr(self.main_dataset, 'mean'):
            if partially_observed and not self.main_dataset.use_mask:
                return (gen_data - self.main_dataset.mean[:, self.input_shape_po:]) \
                                  / ( self.main_dataset.std[:, self.input_shape_po:] + 1e-12 ) 
            else:
                return (gen_data - self.main_dataset.mean) / (self.main_dataset.std + 1e-12) 
        else:
            return gen_data

    def move_from_pool_to_train(self, pool_inds_to_move: Union[int, list, np.ndarray], features_selected: np.array = None,
                                                                    imputation_type: str = 'LLM', mask_inputs = None,
                                                                    po_dumb_imput_not_acq = False) -> None:
        """
        Important:
        - pool_inds_to_move and pool_inds_to_keep index into self.inds["pool"]
        - self.inds["pool"] and train_inds_to_add index into self.main_dataset
        """
        if not isinstance(pool_inds_to_move, (list, np.ndarray)):
            pool_inds_to_move = [pool_inds_to_move]

        pool_inds_to_keep = np.setdiff1d(range(len(self.inds["pool"])), pool_inds_to_move)
        train_inds_to_add = [self.inds["pool"][ind] for ind in pool_inds_to_move]

        self.inds["pool"] = [self.inds["pool"][ind] for ind in pool_inds_to_keep]
        self.inds["train"].extend(train_inds_to_add)
        if po_dumb_imput_not_acq:
            self.main_dataset.modify_data(None, np.array(train_inds_to_add),
                                        imputation_type = imputation_type, mask_inputs = mask_inputs) 
        if features_selected is not None:
            self.main_dataset.modify_data(features_selected[pool_inds_to_move], np.array(train_inds_to_add),
                                        imputation_type = imputation_type, mask_inputs = mask_inputs) 
            
    def replace_nan(self, dataframes):
        # Iterate through each dataframe
        for i, df in enumerate(dataframes):
            # Iterate through each row
            for index, row in df.iterrows():
                # Iterate through each column
                for col in df.columns:
                    # If the value is NaN, replace it with the mode from other dataframes
                    if pd.isna(row[col]):
                        mode_value = None
        
                        # Iterate through other dataframes to find the mode value
                        for j, other_df in enumerate(dataframes):
                            if i != j:  # Skip the current dataframe
                                mode_value_other_df = other_df.at[index, col]
                                if pd.notna(mode_value_other_df):
                                    mode_value = mode_value_other_df
                                    break
        
                        # If mode value is found, replace NaN with mode
                        if mode_value is not None:
                            df.at[index, col] = mode_value




    def load_llm_samples(self, is_numpy = True, get_original = False, filter_unobs = False, apply_mode = True,
                             max_mc_samples = 0, max_mc_samples_random = False) -> np.array:

        # Define a timeout handler
        def handler(signum, frame):
            raise TimeoutError("Reading CSV took too long")

        # Function to safely read CSV within a separate process
        def read_csv_in_process(filepath):
            return pd.read_csv(filepath)

        # Wrapper function to handle timeout for reading CSV
        def read_csv_with_subprocess(filepath, timeout):
            try:
                # Run a subprocess to read the CSV file
                result = subprocess.run(
                    ['python', '-c', f'import pandas as pd; df = pd.read_csv("{filepath}"); print(df.to_csv())'], 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True
                )
                
                if result.returncode != 0:
                    print(f"Error reading {filepath}: {result.stderr}")
                    return None

                # Use the stdout (CSV data as string) to create a DataFrame
                df_data = pd.read_csv(io.StringIO(result.stdout))
                return df_data

            except subprocess.TimeoutExpired:
                print(f"Timeout while reading {filepath}")
                return None
            except Exception as e:
                print(f"Error while processing {filepath}: {e}")
                return None
            
        #signal.signal(signal.SIGALRM, handler)
        all_data   = []
        index_used = self.get_original_index_pool_data()
        for this_dir_pd in os.listdir(self.dir_llm_samples):
            #try:
            #signal.alarm(1)  # 1 second timeout
            #print("read before csv: ", this_dir_pd)
            df_data = pd.read_csv(f'{self.dir_llm_samples}/{this_dir_pd}')
            #df_data = read_csv_with_subprocess(f'{self.dir_llm_samples}/{this_dir_pd}', timeout=1)

            if df_data is None:
                continue

            #print("read after csv: ", df_data.shape)
            #signal.alarm(0)  # Disable the alarm once done

            if 'Unnamed: 0' in df_data.columns:
                df_data = df_data.set_index('Unnamed: 0')

            df_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            # if apply_mode:
            #     df_data = df_data.apply(lambda x: x.fillna(x.mode()[0]))
            df_data = self.main_dataset.update_columns_data(df_data, get_original, filter_unobs)
            try:
                all_data += [df_data.loc[index_used]]
            except:
                continue
            # except TimeoutError:
            #     print(f"Timeout while reading {this_dir_pd}")
            #     continue
            # except Exception as e:
            #     print(f"Error while processing {this_dir_pd}: {e}")
            #     continue

        if apply_mode:
            self.replace_nan(all_data)
        if max_mc_samples == 0:
            all_data_used = all_data
        else:
            if max_mc_samples_random:
                ar = np.arange(len(all_data))
                np.random.shuffle(ar)
                all_data_used = [all_data[idx] for idx in ar]
                all_data_used = all_data_used[:max_mc_samples]
            else:
                all_data_used = all_data[:max_mc_samples]
        if is_numpy:
            return np.stack([this_data.to_numpy() for this_data in all_data_used], axis = 1)        
        else:
            return all_data_used


    def load_cost_data(self, include_y_cost, different_x_cost, stochastic_cost):
        cost_data_np = self.main_dataset.load_cost_data(include_y_cost, different_x_cost, stochastic_cost)
        return cost_data_np
