import bisect
import feather
from itertools import combinations
import numpy as np
import pathlib
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering
from typing import Callable, List

from ark.utils.misc_utils import verify_in_list

###############################################
# Copyright Žiga Sajovic, XLAB 2019           #
# Distributed under the MIT License           #
#                                             #
# github.com/ZigaSajovic/Consensus_Clustering #
#                                             #
###############################################

class ConsensusCluster:
    def __init__(self, cluster: Callable, L: int, K: int, H: int,
                 resample_proportion: float = 0.5):
        """
        Implementation of Consensus clustering, following the paper
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
            self.Ak[i] = np.sum(h*(b-a)
                             for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
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
        """
        Predicts on the data, for best found cluster number
        
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

        self.cluster_type = cluster_type
        self.som_col = '%s_som_cluster' % cluster_type
        self.meta_col = '%s_meta_cluster' % cluster_type
        self.input_data = pd.read_csv(input_file)
        self.columns = columns
        self.max_k = max_k
        self.cap = cap
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
        self.mapping = self.input_data[[self.som_col, self.meta_col]]

    def save_som_to_meta_map(self, save_path: pathlib.Path):
        """Saves the mapping generated by `ConsensusCluster` to `save_path`.

        Args:
            save_path (pathlib.Path):
                The path to save `self.mapping` to.
        """
        self.mapping.to_csv(save_path)

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
