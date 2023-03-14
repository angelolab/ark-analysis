import bisect
import os
import pathlib
import warnings
from abc import ABC, abstractmethod
from itertools import combinations
from typing import List, Protocol, runtime_checkable

import feather
import numpy as np
import pandas as pd
from pyFlowSOM import map_data_to_nodes, som
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering
from alpineer.io_utils import list_files, validate_paths
from alpineer.misc_utils import verify_in_list


class PixieSOMCluster(ABC):
    @abstractmethod
    def __init__(self, weights_path: pathlib.Path, columns: List[str], num_passes: int = 1,
                 xdim: int = 10, ydim: int = 10, lr_start: float = 0.05, lr_end: float = 0.01,
                 seed=42):
        """Generic implementation of a pyFlowSOM runner

        Args:
            weights_path (pathlib.Path):
                The path to save the weights to.
            columns (List[str]):
                The list of columns to subset the data on.
            num_passes (int):
                The number of SOM training passes to use.
            xdim (int):
                The number of SOM nodes on the x-axis.
            ydim (int):
                The number of SOM nodes on the y-axis.
            lr_start (float):
                The initial learning rate.
            lr_end (float):
                The learning rate to decay to
            seed (int):
                The random seed to use for training.
        """
        self.weights_path = weights_path
        self.weights = None if not os.path.exists(weights_path) else feather.read_dataframe(
            weights_path
        )
        self.columns = columns
        self.num_passes = num_passes
        self.xdim = xdim
        self.ydim = ydim
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.seed = seed

    @abstractmethod
    def normalize_data(self) -> pd.DataFrame:
        """Generic implementation of the normalization process to use on the input data

        Returns:
            pandas.DataFrame:
                The data with `columns` normalized by the values in `norm_data`
        """

    def train_som(self, data: pd.DataFrame):
        """Trains the SOM on the data provided and saves the weights generated

        Args:
            data (pandas.DataFrame):
                The input data to train the SOM on.
        """
        # pyFlowSOM.som requires data in np.float64, add type cast for safety purposes
        som_weights = som(
            data=data.values.astype(np.float64), xdim=self.xdim, ydim=self.ydim,
            rlen=self.num_passes, alpha_range=(self.lr_start, self.lr_end), seed=self.seed
        )

        # ensure dimensions of weights are flattened
        som_weights = np.reshape(som_weights, (self.xdim * self.ydim, som_weights.shape[-1]))
        self.weights = pd.DataFrame(som_weights, columns=data.columns.values)

        # save the weights to weights_path
        feather.write_dataframe(self.weights, self.weights_path, compression='uncompressed')

    def generate_som_clusters(self, external_data: pd.DataFrame) -> np.ndarray:
        """Uses the weights to generate SOM clusters for a dataset

        Args:
            external_data (pandas.DataFrame):
                The dataset to generate SOM clusters for

        Returns:
            numpy.ndarray:
                The SOM clusters generated for each pixel in `external_data`
        """
        # subset on just the weights columns prior to SOM cluster mapping
        weights_cols = self.weights.columns.values

        # ensure the weights cols are actually contained in external_data
        verify_in_list(
            weights_cols=weights_cols,
            external_data_cols=external_data.columns.values
        )

        # define the batches of cluster labels assigned
        cluster_labels = []

        # work in batches of 100 to account to support large dataframe sizes
        # TODO: possible dynamic computation in order?
        for i in np.arange(0, external_data.shape[0], 100):
            # NOTE: this also orders the columns of external_data_sub the same as self.weights
            cluster_labels.append(map_data_to_nodes(
                self.weights.values.astype(np.float64),
                external_data.loc[
                    i:min(i + 99, external_data.shape[0]), weights_cols
                ].values.astype(np.float64)
            )[0])

        # concat all the results together and return
        return np.concatenate(cluster_labels)


class PixelSOMCluster(PixieSOMCluster):
    def __init__(self, pixel_subset_folder: pathlib.Path, norm_vals_path: pathlib.Path,
                 weights_path: pathlib.Path, fovs: List[str], columns: List[str],
                 num_passes: int = 1, xdim: int = 10, ydim: int = 10,
                 lr_start: float = 0.05, lr_end: float = 0.01, seed=42):
        """Creates a pixel SOM cluster object derived from the abstract PixieSOMCluster

        Args:
            pixel_subset_folder (pathlib.Path):
                The name of the subsetted pixel data directory
            norm_vals_path (pathlib.Path):
                The name of the feather file containing the normalization values.
            weights_path (pathlib.Path):
                The path to save the weights to.
            fovs (List[str]):
                The list of FOVs to subset the data on.
            columns (List[str]):
                The list of columns to subset the data on.
            num_passes (int):
                The number of SOM training passes to use.
            xdim (int):
                The number of SOM nodes on the x-axis.
            ydim (int):
                The number of SOM nodes on the y-axis.
            lr_start (float):
                The initial learning rate.
            lr_end (float):
                The learning rate to decay to.
            seed (int):
                The random seed to use.
        """
        super().__init__(
            weights_path, columns, num_passes, xdim, ydim, lr_start, lr_end, seed
        )

        # path validation
        validate_paths([norm_vals_path, pixel_subset_folder])

        # load the normalization values in
        self.norm_data = feather.read_dataframe(norm_vals_path)

        # define the fovs used
        self.fovs = fovs

        # list all the files in pixel_subset_folder and load them to train_data
        fov_files = list_files(pixel_subset_folder, substrs='.feather')
        self.train_data = pd.concat(
            [feather.read_dataframe(os.path.join(pixel_subset_folder, fov)) for fov in fov_files
             if os.path.splitext(fov)[0] in fovs]
        )

        # we can just normalize train_data now since that's what we'll be training on
        self.train_data = self.normalize_data(self.train_data)

    def normalize_data(self, external_data: pd.DataFrame) -> pd.DataFrame:
        """Uses `norm_data` to normalize a dataset

        Args:
            external_data (pandas.DataFrame):
                The data to normalize

        Returns:
            pandas.DataFrame:
                The data with `columns` normalized by the values in `norm_data`
        """

        # verify norm_data_cols actually contained in external_data
        verify_in_list(
            norm_data_cols=self.norm_data.columns.values,
            external_data_cols=external_data.columns.values
        )

        # ensure columns in norm_data match up with external_data before normalizing
        norm_data_cols = self.norm_data.columns.values
        external_data_norm = external_data.copy()
        external_data_norm[norm_data_cols] = external_data_norm[norm_data_cols].div(
            self.norm_data.iloc[0], axis=1
        )

        return external_data_norm

    def train_som(self, overwrite=False):
        """Trains the SOM using `train_data`

        overwrite (bool):
            If set, force retrains the SOM and overwrites the weights
        """
        # if overwrite flag set, retrain SOM regardless of state
        if overwrite:
            warnings.warn('Overwrite flag set, retraining SOM')
        # otherwise, do not train SOM if weights already exist and the same markers used to train
        elif self.weights is not None:
            if set(self.weights.columns.values) == set(self.columns):
                warnings.warn('Pixel SOM already trained on specified markers')
                return

            # notify the user that different markers specified
            warnings.warn('New markers specified, retraining')

        super().train_som(self.train_data[self.columns])

    def assign_som_clusters(self, external_data: pd.DataFrame) -> pd.DataFrame:
        """Assigns SOM clusters using `weights` to a dataset

        Args:
            external_data (pandas.DataFrame):
                The dataset to assign SOM clusters to

        Returns:
            pandas.DataFrame:
                The dataset with the SOM clusters assigned.
        """
        # normalize external_data prior to assignment
        external_data_norm = self.normalize_data(external_data)
        som_labels = super().generate_som_clusters(external_data_norm)

        # assign SOM clusters to external_data
        external_data_norm['pixel_som_cluster'] = som_labels

        return external_data_norm


class CellSOMCluster(PixieSOMCluster):
    def __init__(self, cell_data: pd.DataFrame, weights_path: pathlib.Path,
                 fovs: List[str], columns: List[str], num_passes: int = 1,
                 xdim: int = 10, ydim: int = 10, lr_start: float = 0.05, lr_end: float = 0.01,
                 seed=42):
        """Creates a cell SOM cluster object derived from the abstract PixieSOMCluster

        Args:
            cell_data (pandas.DataFrame):
                The dataset to use for training
            weights_path (pathlib.Path):
                The path to save the weights to.
            fovs (List[str]):
                The list of FOVs to subset the data on.
            columns (List[str]):
                The list of columns to subset the data on.
            num_passes (int):
                The number of SOM training passes to use.
            xdim (int):
                The number of SOM nodes on the x-axis.
            ydim (int):
                The number of SOM nodes on the y-axis.
            lr_start (float):
                The initial learning rate.
            lr_end (float):
                The learning rate to decay to.
            seed (int):
                The random seed to use.
        """
        super().__init__(
            weights_path, columns, num_passes, xdim, ydim, lr_start, lr_end, seed
        )

        # assign the cell data
        self.cell_data = cell_data

        # define the fovs used
        self.fovs = fovs

        # subset cell_data on just the FOVs specified
        self.cell_data = self.cell_data[
            self.cell_data['fov'].isin(self.fovs)
        ].reset_index(drop=True)

        # since cell_data is the only dataset, we can just normalize it immediately
        self.normalize_data()

    def normalize_data(self):
        """Normalizes `cell_data` by the 99.9% value of each pixel cluster count column

        Returns:
            pandas.DataFrame:
                `cell_data` with `columns` normalized by the values in `norm_data`
        """
        # only 99.9% normalize on the columns provided
        cell_data_sub = self.cell_data[self.columns].copy()

        # compute the 99.9% normalization values, ignoring zeros
        cell_norm_vals = cell_data_sub.replace(0, np.nan).quantile(q=0.999, axis=0)

        # divide cell_data_sub by normalization values
        cell_data_sub = cell_data_sub.div(cell_norm_vals)

        # assign back to cell_data
        self.cell_data[self.columns] = cell_data_sub

    def train_som(self, overwrite=False):
        """Trains the SOM using `cell_data`

        overwrite (bool):
            If set, force retrains the SOM and overwrites the weights
        """
        # if overwrite flag set, retrain SOM regardless of state
        if overwrite:
            warnings.warn('Overwrite flag set, retraining SOM')

        # otherwise, do not train SOM if weights already exist and the same columns used to train
        elif self.weights is not None:
            if set(self.weights.columns.values) == set(self.columns):
                warnings.warn('Cell SOM already trained on specified columns')
                return

            # notify the user that different columns specified
            warnings.warn('New columns specified, retraining')

        super().train_som(self.cell_data[self.columns])

    def assign_som_clusters(self) -> pd.DataFrame:
        """Assigns SOM clusters using `weights` to `cell_data`

        Args:
            external_data (pandas.DataFrame):
                The dataset to assign SOM clusters to

        Returns:
            pandas.DataFrame:
                `cell_data` with the SOM clusters assigned.
        """
        # cell_data is already normalized, don't repeat
        som_labels = super().generate_som_clusters(self.cell_data[self.columns])

        # assign SOM clusters to cell_data
        self.cell_data['cell_som_cluster'] = som_labels

        return self.cell_data


# define a template class for type hinting cluster param in ConsensusCluster constructor
@runtime_checkable
class ClusterClassTemplate(Protocol):
    def fit_predict(self) -> None:
        ...

    @property
    def n_clusters(self) -> int:
        return n_cluster

###############################################
# Copyright Žiga Sajovic, XLAB 2019           #
# Distributed under the MIT License           #
#                                             #
# github.com/ZigaSajovic/Consensus_Clustering #
#                                             #
###############################################


class ConsensusCluster:
    def __init__(self, cluster: ClusterClassTemplate, L: int, K: int, H: int,
                 resample_proportion: float = 0.5):
        """Implementation of Consensus clustering, following the paper
        https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf

        Args:
            cluster (Callable):
                Clustering class.

                NOTE: the class is to be instantiated with parameter `n_clusters`,
                and possess a `fit_predict` method, which is invoked on data.
            L (int):
                Smallest number of clusters to try.
            K (int):
                Biggest number of clusters to try.
            H (int):
                Number of resamplings for each cluster number.
            resample_proportion (float):
                Percentage to sample.
            Mk (numpy.ndarray):
                Consensus matrices for each k (shape =(K,data.shape[0],data.shape[0])).
                NOTE: every consensus matrix is retained, like specified in the paper.
            Ak (numpy.ndarray):
                Area under CDF for each number of clusters.
                See paper: section 3.3.1. Consensus distribution.
            deltaK (numpy.ndarray):
                Changes in areas under CDF.
                See paper: section 3.3.1. Consensus distribution.
            self.bestK (int):
                Number of clusters that was found to be best.
        """
        assert 0 <= resample_proportion <= 1, "proportion has to be between 0 and 1"
        self.cluster_ = cluster
        self.resample_proportion_ = resample_proportion
        self.L_ = L
        self.K_ = K
        self.H_ = H
        self.Mk = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None

    def _internal_resample(self, data: pd.DataFrame, proportion: float):
        """Implements resampling procedure.

        Args:
            data (pandas.DataFrame):
                The data in `(examples,attributes)` format.
            proportion (float):
                The percentage to sample.
        """
        resampled_indices = np.random.choice(
            range(data.shape[0]), size=int(data.shape[0]*proportion), replace=False)
        return resampled_indices, data[resampled_indices, :]

    def fit(self, data: pd.DataFrame, verbose: bool = False):
        """Fits a consensus matrix for each number of clusters

        Args:
            data (pd.DataFrame):
                The data in `(examples,attributes)` format.
            verbose (bool):
                Should print or not.
        """
        Mk = np.zeros((self.K_-self.L_, data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],)*2)
        for k in range(self.L_, self.K_):  # for each number of clusters
            i_ = k-self.L_
            if verbose:
                print("At k = %d, aka. iteration = %d" % (k, i_))
            for h in range(self.H_):  # resample H times
                if verbose:
                    print("\tAt resampling h = %d, (k = %d)" % (h, k))
                resampled_indices, resample_data = self._internal_resample(
                    data, self.resample_proportion_)
                Mh = self.cluster_(n_clusters=k).fit_predict(resample_data)
                # find indexes of elements from same clusters with bisection
                # on sorted array => this is more efficient than brute force search
                index_mapping = np.array((Mh, resampled_indices)).T
                index_mapping = index_mapping[index_mapping[:, 0].argsort()]
                sorted_ = index_mapping[:, 0]
                id_clusts = index_mapping[:, 1]
                for i in range(k):  # for each cluster
                    ia = bisect.bisect_left(sorted_, i)
                    ib = bisect.bisect_right(sorted_, i)
                    is_ = id_clusts[ia:ib]
                    ids_ = np.array(list(combinations(is_, 2))).T
                    # sometimes only one element is in a cluster (no combinations)
                    if ids_.size != 0:
                        Mk[i_, ids_[0], ids_[1]] += 1
                # increment counts
                ids_2 = np.array(list(combinations(resampled_indices, 2))).T
                Is[ids_2[0], ids_2[1]] += 1
            Mk[i_] /= Is+1e-8  # consensus matrix
            # Mk[i_] is upper triangular (with zeros on diagonal), we now make it symmetric
            Mk[i_] += Mk[i_].T
            Mk[i_, range(data.shape[0]), range(
                data.shape[0])] = 1  # always with self
            Is.fill(0)  # reset counter
        self.Mk = Mk
        # fits areas under the CDFs
        self.Ak = np.zeros(self.K_-self.L_)
        for i, m in enumerate(Mk):
            hist, bins = np.histogram(m.ravel(), density=True)
            self.Ak[i] = np.sum(h*(b-a) for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
        # fits differences between areas under CDFs
        self.deltaK = np.array([(Ab-Aa)/Aa if i > 2 else Aa
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1],
                                                     range(self.L_, self.K_-1))])
        self.bestK = np.argmax(self.deltaK) + \
            self.L_ if self.deltaK.size > 0 else self.L_

    def predict(self):
        """Predicts on the consensus matrix, for best found cluster number.

        Returns:
            numpy.ndarray:
                The consensus matrix prediction for `self.bestK`.
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            1-self.Mk[self.bestK-self.L_])

    def predict_data(self, data: pd.DataFrame):
        """Predicts on the data, for best found cluster number

        Args:
            data (pandas.DataFrame):
                `(examples,attributes)` format

        Returns:
            pandas.DataFrame:
                The data matrix prediction for `self.bestK`.
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            data)


class PixieConsensusCluster:
    def __init__(self, cluster_type: str, input_file: pathlib.Path, columns: List[str],
                 max_k: int = 20, cap: float = 3):
        """Constructs a generic ConsensusCluster pipeline object that makes use of
        Sagovic's implementation of consensus clustering in Python.

        Args:
            cluster_type (str):
                The type of data being run through consensus clustering.
                Must be either `'pixel'` or `'cell'`
            input_file (pathlib.Path):
                The average expression values per SOM cluster .csv,
                computed by `ark.phenotyping.cluster_pixels` or `ark.phenotyping.cluster_cells`
                depending on the type of data being generated.
            columns (List[str]):
                The list of columns to subset the data in `input_file` on for consensus clustering.
            max_k (int):
                The number of consensus clusters to use.
            cap (float):
                The value to cap the data in `input_file` at after z-score normalization.
                Data will be within the range `[-cap, cap]`.
        """
        # validate the cluster_type provided
        verify_in_list(
            provided_cluster_type=cluster_type,
            supported_cluster_types=['pixel', 'cell']
        )

        # path validation
        validate_paths([input_file])

        self.cluster_type = cluster_type
        self.som_col = '%s_som_cluster' % cluster_type
        self.meta_col = '%s_meta_cluster' % cluster_type
        self.input_file = input_file
        self.input_data = pd.read_csv(input_file)
        self.columns = columns
        self.max_k = max_k
        self.cap = cap

        # NOTE: H set to 10 to replicate default 'reps' arg in R ConsensusClusterPlus
        # resample_proportion set to 0.8 to replicate default 'pItem' arg in ConsensusClusterPlus
        self.cc = ConsensusCluster(
            cluster=AgglomerativeClustering,
            L=max_k,
            K=max_k,
            H=10,
            resample_proportion=0.8
        )

        self.mapping = None

    def scale_data(self):
        """z-scores and caps `input_data`.

        Scaling will be done on a per-column basis for all column names specified.
        Capping will truncate the data in the range `[-cap, cap]`.
        """
        # z-score the data
        self.input_data[self.columns] = self.input_data[self.columns].apply(zscore)

        # cap the data in the range [-cap, cap]
        self.input_data[self.columns] = self.input_data[self.columns].clip(
            lower=-self.cap, upper=self.cap
        )

    def run_consensus_clustering(self):
        """Fits the meta clustering results using `ConsensusCluster`.
        """
        self.cc.fit(self.input_data[self.columns])

    def generate_som_to_meta_map(self):
        """Maps each `'{self.cluster_type}_som_cluster'` to the meta cluster
        generated by `ConsensusCluster`.

        Also assigns mapping to `self.mapping` for use in `assign_consensus_labels`.
        """
        self.input_data[self.meta_col] = self.cc.predict_data(self.input_data[self.columns])
        self.mapping = self.input_data[[self.som_col, self.meta_col]].copy()

        # we assume clusters are 1-indexed, so need to correct for Sagovic's 0-indexing
        self.mapping.loc[:, self.meta_col] += 1

    def save_som_to_meta_map(self, save_path: pathlib.Path):
        """Saves the mapping generated by `ConsensusCluster` to `save_path`.

        Args:
            save_path (pathlib.Path):
                The path to save `self.mapping` to.
        """
        feather.write_dataframe(self.mapping, save_path)

    def assign_consensus_labels(self, external_data: pd.DataFrame) -> pd.DataFrame:
        """Takes an external dataset and applies `ConsensusCluster` mapping to it.

        Args:
            external_data (pandas.DataFrame):
                A dataset which contains a `'{self.cluster_type}_som_cluster'` column.

        Returns:
            pandas.DataFrame:
                The `external_data` with a `'{self.cluster_type}_meta_cluster'` column attached.
        """
        external_data[self.meta_col] = external_data[self.som_col].map(
            self.mapping.set_index(self.som_col)[self.meta_col]
        )
        return external_data
