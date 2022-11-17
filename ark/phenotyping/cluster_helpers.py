import bisect
import feather
from itertools import combinations
import numpy as np
import pathlib
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering
from typing import List

from ark.utils.misc_utils import verify_in_list

###############################################
# Copyright Žiga Sajovic, XLAB 2019           #
# Distributed under the MIT License           #
#                                             #
# github.com/ZigaSajovic/Consensus_Clustering #
#                                             #
###############################################


class ConsensusCluster:
    """
      Implementation of Consensus clustering, following the paper
      https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
      Args:
        * cluster -> clustering class
        * NOTE: the class is to be instantiated with parameter `n_clusters`,
          and possess a `fit_predict` method, which is invoked on data.
        * L -> smallest number of clusters to try
        * K -> biggest number of clusters to try
        * H -> number of resamplings for each cluster number
        * resample_proportion -> percentage to sample
        * Mk -> consensus matrices for each k (shape =(K,data.shape[0],data.shape[0]))
                (NOTE: every consensus matrix is retained, like specified in the paper)
        * Ak -> area under CDF for each number of clusters 
                (see paper: section 3.3.1. Consensus distribution.)
        * deltaK -> changes in areas under CDF
                (see paper: section 3.3.1. Consensus distribution.)
        * self.bestK -> number of clusters that was found to be best
      """

    def __init__(self, cluster, L, K, H, resample_proportion=0.5):
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

    def _internal_resample(self, data, proportion):
        """
        Args:
          * data -> (examples,attributes) format
          * proportion -> percentage to sample
        """
        resampled_indices = np.random.choice(
            range(data.shape[0]), size=int(data.shape[0]*proportion), replace=False)
        return resampled_indices, data[resampled_indices, :]

    def fit(self, data, verbose=False):
        """
        Fits a consensus matrix for each number of clusters
        Args:
          * data -> (examples,attributes) format
          * verbose -> should print or not
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
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1], range(self.L_, self.K_-1))])
        self.bestK = np.argmax(self.deltaK) + \
            self.L_ if self.deltaK.size > 0 else self.L_

    def predict(self):
        """
        Predicts on the consensus matrix, for best found cluster number
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            1-self.Mk[self.bestK-self.L_])

    def predict_data(self, data):
        """
        Predicts on the data, for best found cluster number
        Args:
          * data -> (examples,attributes) format 
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            data)


class PixieConsensusCluster:
    def __init__(self, input_file: pathlib.Path, columns: List[str], max_k: int = 20, cap: float = 3):
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
        # equivalent to scale in R
        self.input_data[self.columns] = self.input_data[self.columns].apply(zscore)

        # equivalent to pmin and pmax in R
        self.input_data[self.columns] = self.input_data[self.columns].clip(lower=-self.cap, upper=self.cap)

    def run_consensus_clustering(self):
        self.cc.fit(self.input_data[self.columns])

    def generate_som_to_meta_map(self):
        self.input_data['pixel_meta_cluster'] = self.cc.predict_data(self.input_data[self.columns])

    def save_som_to_meta_map(self, save_path: pathlib.Path):
        self.mapping.to_csv(save_path)

    def assign_consensus_labels(self, cluster_type: str, external_data: pd.DataFrame) -> pd.DataFrame:
        verify_in_list(
            provided_cluster_type=cluster_type,
            supported_cluster_types=['pixel', 'cell']
        )
        som_col = '%s_som_cluster' % cluster_type
        meta_col = '%s_meta_cluster' % cluster_type
        external_data[meta_col] = external_data[som_col].map(self.mapping.set_index(som_col)[meta_col])

        return external_data
