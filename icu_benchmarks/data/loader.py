import warnings
from typing import List
from pandas import DataFrame
import gin
import numpy as np
from torch import Tensor, cat, from_numpy, float32
from torch.utils.data import Dataset
import logging
from typing import Dict, Tuple
import polars as pl
from icu_benchmarks.imputation.amputations import ampute_data
from .constants import DataSegment as Segment
from .constants import DataSplit as Split
# Added together with BATPolarsDataset 
import torch
from torch.nn.functional import pad
from icu_benchmarks.constants import RunMode
# Added together witl SSLPolarsDataset
import random 

import time # [DEBUG]
import os #[ DEBUG]
@gin.configurable("CommonPolarsDataset")
class CommonPolarsDataset(Dataset):
    def __init__(
            self,
            data: dict,
            split: str = Split.train,
            vars: Dict[str, str] = gin.REQUIRED,
            grouping_segment: str = Segment.outcome,
            mps: bool = False,
            name: str = "",
            *args,
            **kwargs,
    ):
        # super().__init__(*args, **kwargs)
        self.split = split
        self.vars = vars
        self.grouping_df = data[split][grouping_segment]  # .set_index(self.vars["GROUP"])
        # logging.info(f"data split: {data[split]}")
        # self.features_df = (
        #     data[split][Segment.features].set_index(self.vars["GROUP"]).drop(labels=self.vars["SEQUENCE"], axis=1)
        # )
        # Get the row indicators for the data to be able to match predicted labels
        if "SEQUENCE" in self.vars and self.vars["SEQUENCE"] in data[split][Segment.features].columns:
            # We have a time series dataset
            self.row_indicators = data[split][Segment.features][self.vars["GROUP"], self.vars["SEQUENCE"]]
            self.row_indicators = self.row_indicators.with_columns(pl.col(self.vars["SEQUENCE"]).dt.total_hours())
            self.features_df = data[split][Segment.features]
            self.features_df = self.features_df.sort([self.vars["GROUP"], self.vars["SEQUENCE"]])
            #self.features_df = self.features_df.drop(self.vars["SEQUENCE"])
        else:
            # We have a static dataset
            logging.info("Using static dataset")
            self.row_indicators = data[split][Segment.features][self.vars["GROUP"]]
            self.features_df = data[split][Segment.features]
        # calculate basic info for the data
        self.num_stays = self.grouping_df[self.vars["GROUP"]].unique().shape[0]
        self.maxlen = self.features_df.group_by([self.vars["GROUP"]]).len().max().item(0, 1)
        self.mps = mps
        self.name = name

    def ram_cache(self, cache: bool = True):
        print(f"[DEBUG] ram_cache() called with cache={cache}")
        self._cached_dataset = None
        if cache:
            logging.info(f"Caching {self.split} dataset in ram.")
            self._cached_dataset = [self[i] for i in range(len(self))]

    def __len__(self) -> int:
        """Returns number of stays in the data.

        Returns:
            number of stays in the data
        """
        return self.num_stays

    def get_feature_names(self) -> List[str]:
        return self.features_df.columns

    def to_tensor(self) -> List[Tensor]:
        values = []
        for entry in self:
            for i, value in enumerate(entry):
                if len(values) <= i:
                    values.append([])
                values[i].append(value.unsqueeze(0))
        return [cat(value, dim=0) for value in values]


@gin.configurable("PredictionPolarsDataset")
class PredictionPolarsDataset(CommonPolarsDataset):
    """Subclass of common dataset for prediction tasks.

    Args:
        ram_cache (bool, optional): Whether the complete dataset should be stored in ram. Defaults to True.
    """

    def __init__(self, *args, ram_cache: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcome_df = self.grouping_df
        self.ram_cache(ram_cache)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice. Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]

        pad_value = 0.0
        # stay_id = self.outcome_df.index.unique()[idx]  # [self.vars["GROUP"]]
        stay_id = self.outcome_df[self.vars["GROUP"]].unique()[idx]  # [self.vars["GROUP"]]

        # slice to make sure to always return a DF
        # window = self.features_df.loc[stay_id:stay_id].to_numpy()
        # labels = self.outcome_df.loc[stay_id:stay_id][self.vars["LABEL"]].to_numpy(dtype=float)
        window = self.features_df.filter(pl.col(self.vars["GROUP"]) == stay_id).select(
            pl.exclude(self.vars["GROUP"])).to_numpy()
        labels = self.outcome_df.filter(pl.col(self.vars["GROUP"]) == stay_id)[self.vars["LABEL"]].to_numpy().astype(
            float)

        if len(labels) == 1:
            # only one label per stay, align with window
            labels = np.concatenate([np.empty(window.shape[0] - 1) * np.nan, labels], axis=0)

        length_diff = self.maxlen - window.shape[0]
        pad_mask = np.ones(window.shape[0])

        # Padding the array to fulfill size requirement
        if length_diff > 0:
            # window shorter than the longest window in dataset, pad to same length
            window = np.concatenate([window, np.ones((length_diff, window.shape[1])) * pad_value], axis=0)
            labels = np.concatenate([labels, np.ones(length_diff) * pad_value], axis=0)
            pad_mask = np.concatenate([pad_mask, np.zeros(length_diff)], axis=0)

        not_labeled = np.argwhere(np.isnan(labels))
        if len(not_labeled) > 0:
            labels[not_labeled] = -1
            pad_mask[not_labeled] = 0

        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        data = window.astype(np.float32)

        return from_numpy(data), from_numpy(labels), from_numpy(pad_mask)

    def get_balance(self) -> list:
        """Return the weight balance for the split of interest.

        Returns:
            Weights for each label.
        """
        counts = self.outcome_df[self.vars["LABEL"]].value_counts(parallel=True).get_columns()[1]
        counts = counts.to_numpy()
        weights = list((1 / counts) * np.sum(counts) / counts.shape[0])
        return weights

    def get_data_and_labels(self) -> Tuple[np.array, np.array, np.array]:
        """Function to return all the data and labels aligned at once.

        We use this function for the ML methods which don't require an iterator.

        Returns:
            A Tuple containing data points and label for the split.
        """
        labels = self.outcome_df[self.vars["LABEL"]].to_numpy().astype(float)
        rep = self.features_df

        if len(labels) == self.num_stays:
            # order of groups could be random, we make sure not to change it
            # rep = rep.groupby(level=self.vars["GROUP"], sort=False).last()
            rep = rep.group_by(self.vars["GROUP"]).last()
        else:
            # Adding segment count for each stay id and timestep.
            rep = rep.with_columns(pl.col(self.vars["GROUP"]).cum_count().over(self.vars["GROUP"]).alias("counter"))
        rep = rep.to_numpy().astype(float)
        logging.debug(f"rep shape: {rep.shape}")
        logging.debug(f"labels shape: {labels.shape}")
        return rep, labels, self.row_indicators.to_numpy()

    def to_tensor(self) -> Tuple[Tensor, Tensor, Tensor]:
        data, labels, row_indicators = self.get_data_and_labels()
        if self.mps:
            return from_numpy(data).to(float32), from_numpy(labels).to(float32), from_numpy(row_indicators).to(float32)
        else:
            return from_numpy(data), from_numpy(labels), row_indicators


@gin.configurable("CommonPandasDataset")
class CommonPandasDataset(Dataset):
    """Common dataset: subclass of Torch Dataset that represents the data to learn on.

    Args: data: Dict of the different splits of the data. split: Either 'train','val' or 'test'. vars: Contains the names of
    columns in the data. grouping_segment: str, optional: The segment of the data contains the grouping column with only
    unique values. Defaults to Segment.outcome. Is used to calculate the number of stays in the data.
    """

    def __init__(
            self,
            data: dict,
            split: str = Split.train,
            vars: Dict[str, str] = gin.REQUIRED,
            grouping_segment: str = Segment.outcome,
            mps: bool = False,
            name: str = "",
    ):
        warnings.warn("CommonPandasDataset is deprecated. Use CommonPolarsDataset instead.", DeprecationWarning,
                      stacklevel=2)
        self.split = split
        self.vars = vars
        self.grouping_df = data[split][grouping_segment].set_index(self.vars["GROUP"])
        # logging.info(f"data split: {data[split]}")
        self.features_df = (
            data[split][Segment.features].set_index(self.vars["GROUP"]).drop(labels=self.vars["SEQUENCE"], axis=1)
        )

        # calculate basic info for the data
        self.num_stays = self.grouping_df.index.unique().shape[0]
        self.maxlen = self.features_df.groupby([self.vars["GROUP"]]).size().max()
        self.mps = mps
        self.name = name

    def ram_cache(self, cache: bool = True):
        self._cached_dataset = None
        if cache:
            logging.info(f"Caching {self.split} dataset in ram.")
            self._cached_dataset = [self[i] for i in range(len(self))]

    def __len__(self) -> int:
        """Returns number of stays in the data.

        Returns:
            number of stays in the data
        """
        return self.num_stays

    def get_feature_names(self) -> List[str]:
        return self.features_df.columns

    def to_tensor(self) -> List[Tensor]:
        values = []
        for entry in self:
            for i, value in enumerate(entry):
                if len(values) <= i:
                    values.append([])
                values[i].append(value.unsqueeze(0))
        return [cat(value, dim=0) for value in values]


@gin.configurable("PredictionPandasDataset")
class PredictionPandasDataset(CommonPandasDataset):
    """Subclass of common dataset for prediction tasks.

    Args:
        ram_cache (bool, optional): Whether the complete dataset should be stored in ram. Defaults to True.
    """

    def __init__(self, *args, ram_cache: bool = True, **kwargs):
        super().__init__(*args, grouping_segment=Segment.outcome, **kwargs)
        self.outcome_df = self.grouping_df
        self.ram_cache(ram_cache)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice. Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]

        pad_value = 0.0
        stay_id = self.outcome_df.index.unique()[idx]  # [self.vars["GROUP"]]

        # slice to make sure to always return a DF
        window = self.features_df.loc[stay_id:stay_id].to_numpy()
        labels = self.outcome_df.loc[stay_id:stay_id][self.vars["LABEL"]].to_numpy(dtype=float)

        if len(labels) == 1:
            # only one label per stay, align with window
            labels = np.concatenate([np.empty(window.shape[0] - 1) * np.nan, labels], axis=0)

        length_diff = self.maxlen - window.shape[0]
        pad_mask = np.ones(window.shape[0])

        # Padding the array to fulfill size requirement
        if length_diff > 0:
            # window shorter than the longest window in dataset, pad to same length
            window = np.concatenate([window, np.ones((length_diff, window.shape[1])) * pad_value], axis=0)
            labels = np.concatenate([labels, np.ones(length_diff) * pad_value], axis=0)
            pad_mask = np.concatenate([pad_mask, np.zeros(length_diff)], axis=0)

        not_labeled = np.argwhere(np.isnan(labels))
        if len(not_labeled) > 0:
            labels[not_labeled] = -1
            pad_mask[not_labeled] = 0

        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        data = window.astype(np.float32)

        return from_numpy(data), from_numpy(labels), from_numpy(pad_mask)

    def get_balance(self) -> list:
        """Return the weight balance for the split of interest.

        Returns:
            Weights for each label.
        """
        counts = self.outcome_df[self.vars["LABEL"]].value_counts()
        # weights = list((1 / counts) * np.sum(counts) / counts.shape[0])
        return list((1 / counts) * np.sum(counts) / counts.shape[0])

    def get_data_and_labels(self) -> Tuple[np.array, np.array]:
        """Function to return all the data and labels aligned at once.

        We use this function for the ML methods which don't require an iterator.

        Returns:
            A Tuple containing data points and label for the split.
        """
        labels = self.outcome_df[self.vars["LABEL"]].to_numpy().astype(float)
        rep = self.features_df
        if len(labels) == self.num_stays:
            # order of groups could be random, we make sure not to change it
            rep = rep.groupby(level=self.vars["GROUP"], sort=False).last()
        rep = rep.to_numpy().astype(float)

        return rep, labels

    def to_tensor(self):
        data, labels = self.get_data_and_labels()
        if self.mps:
            return from_numpy(data).to(float32), from_numpy(labels).to(float32)
        else:
            return from_numpy(data), from_numpy(labels)


@gin.configurable("ImputationPandasDataset")
class ImputationPandasDataset(CommonPandasDataset):
    """Subclass of Common Dataset that contains data for imputation models."""

    def __init__(
            self,
            data: Dict[str, DataFrame],
            split: str = Split.train,
            vars: Dict[str, str] = gin.REQUIRED,
            mask_proportion=0.3,
            mask_method="MCAR",
            mask_observation_proportion=0.3,
            ram_cache: bool = True,
    ):
        """
        Args:
            data (Dict[str, DataFrame]): data to use
            split (str, optional): split to apply. Defaults to Split.train.
            vars (Dict[str, str], optional): contains names of columns in the data. Defaults to gin.REQUIRED.
            mask_proportion (float, optional): proportion to artificially mask for amputation. Defaults to 0.3.
            mask_method (str, optional): masking mechanism. Defaults to "MCAR".
            mask_observation_proportion (float, optional): poportion of the observed data to be masked. Defaults to 0.3.
            ram_cache (bool, optional): if the dataset should be completely stored in ram and not generated on the fly during
                training. Defaults to True.
        """
        super().__init__(data, split, vars, grouping_segment=Segment.static)
        self.amputated_values, self.amputation_mask = ampute_data(
            self.features_df, mask_method, mask_proportion, mask_observation_proportion
        )
        self.amputation_mask = (self.amputation_mask + self.features_df.isna().values).bool()
        self.amputation_mask = DataFrame(self.amputation_mask, columns=self.vars[Segment.dynamic])
        self.amputation_mask[self.vars["GROUP"]] = self.features_df.index
        self.amputation_mask.set_index(self.vars["GROUP"], inplace=True)

        self.target_missingness_mask = self.features_df.isna()
        self.features_df.fillna(0, inplace=True)
        self.ram_cache(ram_cache)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice.

        Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]
        stay_id = self.grouping_df.iloc[idx].name

        # slice to make sure to always return a DF
        window = self.features_df.loc[stay_id:stay_id, self.vars[Segment.dynamic]]
        window_missingness_mask = self.target_missingness_mask.loc[stay_id:stay_id, self.vars[Segment.dynamic]]
        amputated_window = self.amputated_values.loc[stay_id:stay_id, self.vars[Segment.dynamic]]
        amputation_mask = self.amputation_mask.loc[stay_id:stay_id, self.vars[Segment.dynamic]]

        return (
            from_numpy(amputated_window.values).to(float32),
            from_numpy(amputation_mask.values).to(float32),
            from_numpy(window.values).to(float32),
            from_numpy(window_missingness_mask.values).to(float32),
        )


@gin.configurable("ImputationPredictionDataset")
class ImputationPredictionDataset(Dataset):
    """Subclass of torch dataset that represents data with missingness for imputation.

    Args:
        data (DataFrame): dict of the different splits of the data
        grouping_column (str, optional): column that is used for grouping. Defaults to "stay_id".
        select_columns (List[str], optional): the columns to serve as input for the imputation model. Defaults to None.
        ram_cache (bool, optional): wether the dataset should be stored in ram. Defaults to True.
    """

    def __init__(
            self,
            data: DataFrame,
            grouping_column: str = "stay_id",
            select_columns: List[str] = None,
            ram_cache: bool = True,
    ):
        self.dyn_df = data

        if select_columns is not None:
            self.dyn_df = self.dyn_df[list(select_columns) + grouping_column]

        if grouping_column is not None:
            self.dyn_df = self.dyn_df.set_index(grouping_column)
        else:
            self.dyn_df = data

        # calculate basic info for the data
        self.group_indices = self.dyn_df.index.unique()
        self.maxlen = self.dyn_df.groupby(grouping_column).size().max()

        self._cached_dataset = None
        if ram_cache:
            logging.info("Caching dataset in ram.")
            self._cached_dataset = [self[i] for i in range(len(self))]

    def __len__(self) -> int:
        """Returns number of stays in the data.

        Returns:
            number of stays in the data
        """
        return self.group_indices.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice.

        Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]
        stay_id = self.group_indices[idx]

        # slice to make sure to always return a DF
        window = self.dyn_df.loc[stay_id:stay_id, :]

        return from_numpy(window.values).to(float32)

class PadToLongestCollator:
    def __init__(self, runmode, pad_1d_tensor, pad_2d_tensor):
        self.runmode = runmode
        self.pad_1d_tensor = pad_1d_tensor
        self.pad_2d_tensor = pad_2d_tensor

    def __call__(self, batch):
        #print(f"[DEBUG] PadToLongestCollator called in pid={os.getpid()}, batch size={len(batch)}")

        data, mask, labels, times, static, delta = zip(*batch)
        #print("[DEBUG] Unpacked batch")

        max_len = max(x.shape[-1] for x in data)
        orig_lens = [x.shape[-1] for x in data]
        #print(f"[DEBUG] max_len={max_len}")

        data = torch.stack([self.pad_2d_tensor(x, max_len) for x in data])
        #print("[DEBUG] data padded")

        mask = torch.stack([self.pad_2d_tensor(x, max_len) for x in mask])
        delta = torch.stack([self.pad_2d_tensor(x, max_len) for x in delta])
        #print("[DEBUG] mask and delta padded")

        times = torch.stack([self.pad_1d_tensor(x, max_len) for x in times])
        static = torch.stack(static)
        #print("[DEBUG] times and static padded")

        if self.runmode == "regression":
            labels = torch.stack([self.pad_1d_tensor(x, max_len) for x in labels])
        else:
            labels = torch.stack(labels).squeeze()
        #print("[DEBUG] labels padded")

        obs_mask = torch.zeros((len(data), max_len), dtype=torch.int32)
        for i, l in enumerate(orig_lens):
            obs_mask[i, :l] = 1
        #print("[DEBUG] obs_mask computed")

        return data, mask, labels, times, static, delta, obs_mask


@gin.configurable("BATPolarsDataset")
class BATPolarsDataset(CommonPolarsDataset):
    """Subclass of common dataset for prediction tasks.

    Args:
        ram_cache (bool, optional): Whether the complete dataset should be stored in ram. Defaults to True.
    """

    def __init__(self, *args, ram_cache: bool = True, runmode=None, vars: Dict[str, str] = gin.REQUIRED, **kwargs):
        super().__init__(vars=vars, *args, **kwargs)
        self.outcome_df = self.grouping_df
        self.runmode = runmode
        self._stay_ids = self.outcome_df[self.vars["GROUP"]].unique().to_numpy()


        #print(f"[DEBUG] BATPolarsDataset __init__: ram_cache={ram_cache}")
        group_col = self.vars["GROUP"]
        #dynamic_columns = self.vars["DYNAMIC"]

        grouped = self.features_df.group_by(group_col, maintain_order=True)
        #self.features_by_id = {
        #    stay_id[0] if isinstance(stay_id, tuple) else stay_id: df.drop(group_col)
        #    for stay_id, df in grouped
        #    }
        #self.features_by_id = {
        #    stay_id[0] if isinstance(stay_id, tuple) else stay_id:
        #        df.select(dynamic_columns).to_numpy()
        #    for stay_id, df in grouped
        #    }
        grouped_outcomes = self.outcome_df.group_by(group_col, maintain_order=True)
        self.outcomes_by_id = {
            stay_id[0] if isinstance(stay_id, tuple) else stay_id: df[self.vars["LABEL"]].to_numpy()
            for stay_id, df in grouped_outcomes
            }

        static_columns = self.vars["STATIC"]
        dynamic_columns = self.vars["DYNAMIC"]

        self.dynamic_columns = dynamic_columns
        self.missingness_columns = [f"MissingIndicator_{col}" for col in dynamic_columns]
        self.static_columns = static_columns
        self.sequence_column = self.vars["SEQUENCE"]

        all_columns = (
            self.dynamic_columns
            + self.missingness_columns
            + self.static_columns
            + [self.sequence_column]
        )

        self.column_index_map = {col: i for i, col in enumerate(all_columns)}

        self.features_by_id = {
            int(stay_id[0] if isinstance(stay_id, tuple) else stay_id): df.select(all_columns).to_numpy()
            for stay_id, df in grouped
            }
        #print(f"[DEBUG] features_by_id keys: {len(self.features_by_id)}")
        #print(f"[DEBUG] outcomes_by_id keys: {len(self.outcomes_by_id)}")
        self.ram_cache(ram_cache)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Function to sample from the data split of choice. Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels, padding mask, and other arrays.
        """
        #print(f"[DEBUG] __getitem__ called with idx={idx}, pid={os.getpid()}")

        #print(f"[DEBUG] __getitem__ index: {idx}")
        
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]

        # UNCOMMENT FOR NORMALIZATION OF TIME For normalizing times array (has not been done in preprocessing) 
        #global_max_time_ms = (
        #self.features_df.select(pl.col(self.vars["SEQUENCE"]).max()).item().total_seconds() * 1000  # Convert to milliseconds
        #)
        
        # Extracting the stay_id for the specific index
        #stay_id = self.outcome_df[self.vars["GROUP"]].unique()[idx] 
        stay_id = self._stay_ids[idx] 
        #print(f"[DEBUG] stay_id resolved: {stay_id}")
        #group_col = self.vars["GROUP"]

        #print(f"[DEBUG] Requested stay_id: {stay_id} (type: {type(stay_id)})")
        #print(f"[DEBUG] Sample key from features_by_id: {next(iter(self.features_by_id))} (type: {type(next(iter(self.features_by_id)))})")

        #if stay_id not in self.features_by_id:
        #    print(f"[ERROR] stay_id {stay_id} not found in features_by_id keys!")
        #features = self.features_by_id[stay_id]
        features = self.features_by_id[stay_id]
        col_idx = self.column_index_map
        #print(f"[DEBUG] features fetched")
        # Selecting label column 

        #labels = self.outcome_df.filter(pl.col(self.vars["GROUP"]) == stay_id)[self.vars["LABEL"]].to_numpy()
        labels = self.outcomes_by_id[stay_id]
        #print(f"[DEBUG] labels fetched")
        
        # Select dynamic values (excluding stay_id and time columns)
        #dynamic_columns = self.vars["DYNAMIC"]  # Use DYNAMIC columns defined in gin

        #data = self.features_df.filter(pl.col(self.vars["GROUP"]) == stay_id).select(dynamic_columns).to_numpy()
        
        #data = features.select(dynamic_columns).to_numpy()
        #data = self.features_by_id[stay_id]
        data = features[:, [col_idx[col] for col in self.dynamic_columns]]
        #print(f"[DEBUG] dynamic data extracted")

        #dynamic_columns = self.vars["DYNAMIC"] # tmp it is in init find a way, maybe self 
        # Select missingness indicators for dynamic features (matching the same order as the dynamic features)
        #missingness_columns = [f'MissingIndicator_{col}' for col in dynamic_columns]
        #mask = self.features_df.filter(pl.col(self.vars["GROUP"]) == stay_id).select(missingness_columns).to_numpy()
        #mask = features.select(missingness_columns).to_numpy()
        #mask = (1 - mask)
        mask = 1 - features[:, [col_idx[col] for col in self.missingness_columns]]
        #print(f"[DEBUG] mask computed")

        # Select static features (assuming they are labeled 'age', 'sex', etc. in the dataset)
        #static_columns = self.vars["STATIC"]  # Use STATIC columns defined in gin
        #static = self.features_df.filter(pl.col(self.vars["GROUP"]) == stay_id).select(static_columns)[0].to_numpy().flatten()
        #static = features.select(static_columns)[0].to_numpy().flatten()
        static = features[0, [col_idx[col] for col in self.static_columns]]
        #print(f"[DEBUG] static extracted")

        # Select timeseries 
        #time_column = self.vars["SEQUENCE"]
        #times_raw = self.features_df.filter(pl.col(self.vars["GROUP"]) == stay_id).select(time_column).to_numpy().flatten()
        #times_raw = features.select(time_column).to_numpy().flatten()
        times_raw = features[:, col_idx[self.sequence_column]]
        # tmp solution for times normalization, should be done in preprocessing. See how R did it 
        times_numeric = (times_raw - times_raw[0]).astype('timedelta64[ms]').astype(np.float32)
        # NO NORNALIZATION: Convert milliseconds to minutes
        times = times_numeric / 60000  # 1 minute = 60,000 ms
        # NORMALIZATION: to [-1, 1] based on global max in milliseconds
        #times = (times_numeric / global_max_time_ms) * 2 - 1
        #print(f"[DEBUG] times computed")

        # Array containing delta time values 
        delta = self.get_delta_t(times, data, mask) 
        #print(f"[DEBUG] delta computed")

        # Permute 
        data = data.T
        mask = mask.T
        delta = delta.T

        # Return all of the required arrays as tensors
        return (
            torch.from_numpy(data),
            torch.from_numpy(mask),
            torch.from_numpy(labels),
            torch.from_numpy(times),
            torch.from_numpy(static),
            torch.from_numpy(delta)
            )
    
    @staticmethod
    def get_delta_t(times, measurements, measurement_indicators):
        """
        From R's repo 
        Creates array with time difference from the most recent feature measurement.
        """
        dt_list = []

        # First observation has delta t = 0
        first_dt = np.zeros(measurement_indicators.shape[1:], dtype=np.float32)  # (F,)
        dt_list.append(first_dt)

        last_dt = first_dt.copy()  # Initialize last_dt before the loop
        for i in range(1, measurement_indicators.shape[0]):
            # Calculate time difference only for observed values
            last_dt = np.where(
                measurement_indicators[i - 1],  # If the previous value was observed
                np.full_like(last_dt, times[i] - times[i - 1]),  # Compute time difference
                times[i] - times[i - 1] + last_dt,  # If the previous value was missing, propagate the last valid time difference
            )
            dt_list.append(last_dt)

        dt_array = np.stack(dt_list)  # Combine the list of deltas into a single array
        dt_array = dt_array.astype(np.float32)  # Ensure consistent data type
        dt_array.shape = measurements.shape  # Reshape to match measurements
        dt_array = dt_array * ~(measurement_indicators.astype(bool))  # Mask the missing values

        return dt_array

    @staticmethod
    def pad_2d_tensor(tensor, max_len):
                    pad_amt = max_len - tensor.shape[-1]
                    return pad(tensor, (0, pad_amt)) if pad_amt > 0 else tensor
    @staticmethod           
    def pad_1d_tensor(tensor, max_len):
        pad_amt = max_len - tensor.shape[0]
        return pad(tensor, (0, pad_amt)) if pad_amt > 0 else tensor

    def collate_fn_pad_to_longest_in_batch(self):
        return PadToLongestCollator(
        self.runmode,
        pad_1d_tensor=BATPolarsDataset.pad_1d_tensor,
        pad_2d_tensor=BATPolarsDataset.pad_2d_tensor
    )


    """
    def collate_fn_pad_to_longest_in_batch(self):
        
        #Returns a collate function that pads variable-length time series
        #in a batch to the length of the longest sequence.

        #Returns:
            #Callable: A function that pads and stacks:
                #- data, mask, delta: (B, F, T)
                #- times: (B, T)
                #- static: (B, S)
                3- label: (B,) or (B, T) depending on task
                #- obs_mask: (B, T) indicating valid timesteps
        
        def collate_fn(batch):

            data, mask, labels, times, static, delta = zip(*batch)
            max_len = max(x.shape[-1] for x in data)
            original_lengths = [x.shape[-1] for x in data]

            data   = torch.stack([self.pad_2d_tensor(x, max_len) for x in data])
            mask   = torch.stack([self.pad_2d_tensor(x, max_len) for x in mask])
            delta  = torch.stack([self.pad_2d_tensor(x, max_len) for x in delta])
            times  = torch.stack([self.pad_1d_tensor(x, max_len) for x in times])
            static = torch.stack(static)

            # In a regression setting there is one label per time bin 
            if self.runmode == "regression":
                labels = torch.stack([self.pad_1d_tensor(x, max_len) for x in labels])
            # In a classification setting there is one label per patient/stay_id 
            else:
                labels = torch.stack(labels).squeeze()   

            obs_mask = torch.zeros((len(data), max_len), dtype=torch.int32)
            for i, seq_len in enumerate(original_lengths):
                obs_mask[i, :seq_len] = 1

            return data, mask, labels, times, static, delta, obs_mask

        return collate_fn
    """
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return self.outcome_df[self.vars["GROUP"]].n_unique()
    
    def get_balance(self) -> list:
            """Return the weight balance for the split of interest.

            Returns:
                Weights for each label.
            """
            counts = self.outcome_df[self.vars["LABEL"]].value_counts(parallel=True).get_columns()[1]
            counts = counts.to_numpy()
            weights = list((1 / counts) * np.sum(counts) / counts.shape[0])
            return weights
    
"""
@gin.configurable("SSLPolarsDataset")
class SSLPolarsDataset(BATPolarsDataset):
    def __init__(self, *args, max_obs=24, forecast_horizon=2, runmode=None, **kwargs):
        
        #SSL dataset that slices each batch into observation and forecasting windows.

        #Args:
        #    max_obs (int): Length of the observation window in time bins (e.g., 24 = 24h).
        #    forecast_horizon (int): Length of the forecasting window in time bins (e.g., 2 = 2h).
        
        super().__init__(*args, runmode=runmode, **kwargs)
        self.max_obs = max_obs
        self.forecast_horizon = forecast_horizon

    def collate_fn_ssl_windows(self):
        base_collate = super().collate_fn_pad_to_longest_in_batch()

        def collate_fn(batch):
            data, mask, label, times, static, delta, obs_mask = base_collate(batch)

            #print(f'\n \n \n DEBUG TIMES SHAPE IN BATCH: {times.shape} \n \n \n')
            B, C, T = data.shape

            t1_ix = None
            tries = 0
            max_tries = B  # Retry up to one attempt per patient in the batch

            while t1_ix is None and tries < max_tries:
                patient_idx = random.randint(0, B - 1)
                patient_mask = obs_mask[patient_idx].bool()

                valid_indices = torch.where(patient_mask)[0]

                # Enforce: minimum 12 time bins of history
                valid_indices = valid_indices[valid_indices >= 12]

                # Make sure there's at least one candidate to compute with
                if len(valid_indices) == 0:
                    tries += 1
                    continue

                # Enforce: space for forecast_horizon
                max_index = valid_indices[-1]
                valid_indices = valid_indices[valid_indices <= max_index - self.forecast_horizon]

                if len(valid_indices) == 0:
                    tries += 1
                    continue

                t1_ix = int(np.random.choice(valid_indices.cpu().numpy()))
                break


            if t1_ix is None:
                raise ValueError("No valid t1 index found in batch after retrying.")

            t0_ix = max(0, t1_ix - self.max_obs)
            t2_ix = t1_ix + self.forecast_horizon

            # Slice all patients at same window
            obs_data = data[:, :, t0_ix:t1_ix]
            obs_mask_out = mask[:, :, t0_ix:t1_ix]
            obs_times = times[:, t0_ix:t1_ix]
            obs_delta = delta[:, :, t0_ix:t1_ix]

            forecast_target = data[:, :, t1_ix:t2_ix]
            forecast_mask = mask[:, :, t1_ix:t2_ix]

            #return {
            #        'obs_data': obs_data,
            #        'obs_mask': obs_mask_out,
            #        'obs_times': obs_times,
            #        'obs_delta': obs_delta,
            #        'forecast_target': forecast_target,
            #        'forecast_mask': forecast_mask,
            #        'static': static,
                    #'debug': {  # UNCOMMENT FOR DEBUGGING
                    #    't0_ix': t0_ix,
                    #    't1_ix': t1_ix,
                    #    't2_ix': t2_ix
                    #    }
            #        }

            return (
                obs_data,
                obs_mask_out,
                obs_times,
                obs_delta,
                forecast_target,
                forecast_mask,
                static,
                )
        return collate_fn
"""
class SSLBatchCollator:
    def __init__(self, base_collate, max_obs, forecast_horizon):
        self.base_collate = base_collate
        self.max_obs = max_obs
        self.forecast_horizon = forecast_horizon

    def __call__(self, batch):
        #print(f"[DEBUG] SSLBatchCollator called in pid={os.getpid()}, batch size={len(batch)}")
        data, mask, label, times, static, delta, obs_mask = self.base_collate(batch)
        B, C, T = data.shape

        t1_ix = None
        tries = 0
        max_tries = B

        while t1_ix is None and tries < max_tries:
            idx = torch.randint(0, B, (1,)).item()
            #print(f"[DEBUG] Try #{tries}: patient={idx}")
            valid_idx = torch.where(obs_mask[idx].bool())[0]
            valid_idx = valid_idx[valid_idx >= 12]
            if len(valid_idx) == 0:
                tries += 1
                continue

            max_index = valid_idx[-1]
            valid_idx = valid_idx[valid_idx <= max_index - self.forecast_horizon]
            if len(valid_idx) == 0:
                tries += 1
                continue

            t1_ix = int(np.random.choice(valid_idx.cpu().numpy()))

        if t1_ix is None:
            raise ValueError("No valid t1 index found in batch after retrying.")

        t0_ix = max(0, t1_ix - self.max_obs)
        t2_ix = t1_ix + self.forecast_horizon

        return (
            data[:, :, t0_ix:t1_ix],
            mask[:, :, t0_ix:t1_ix],
            times[:, t0_ix:t1_ix],
            delta[:, :, t0_ix:t1_ix],
            data[:, :, t1_ix:t2_ix],
            mask[:, :, t1_ix:t2_ix],
            static,
        )

@gin.configurable("SSLPolarsDataset")
class SSLPolarsDataset(BATPolarsDataset):
    def __init__(self, *args, max_obs=24, forecast_horizon=2, runmode=None, **kwargs):
        
        #SSL dataset that slices each batch into observation and forecasting windows.

        #Args:
        #    max_obs (int): Length of the observation window in time bins (e.g., 24 = 24h).
        #    forecast_horizon (int): Length of the forecasting window in time bins (e.g., 2 = 2h).
        
        super().__init__(*args, runmode=runmode, **kwargs)
        self.max_obs = max_obs
        self.forecast_horizon = forecast_horizon

    def collate_fn_ssl_windows(self):
        base_collate = super().collate_fn_pad_to_longest_in_batch()
        return SSLBatchCollator(base_collate, self.max_obs, self.forecast_horizon)


