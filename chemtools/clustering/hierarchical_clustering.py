import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import optimal_leaf_ordering, linkage as scipy_linkage
import sys
from typing import List, Optional, Dict, Tuple, Union, Set, Any
import heapq
from chemtools.preprocessing.distance_matrix import DistanceMatrix


class HierarchicalClustering:
    """
    Performs Agglomerative Hierarchical Clustering.

    This class builds a hierarchy of clusters from the bottom up, starting
    with individual data points as separate clusters and merging them based on
    distance. It can use a precomputed distance matrix or calculate one from data.

    Attributes:
        n_samples (int): Number of data points.
        linkage_method (str): Linkage criterion ('single', 'complete', 'average').
        linkage_matrix_ (np.ndarray): SciPy-compatible linkage matrix (Z).
        children_ (np.ndarray): Cluster merge children (for sklearn compatibility).
        distances_ (np.ndarray): Distances at which merges occurred.
        labels_ (np.ndarray): Cluster labels for each sample after fitting.

    References:
        - https://en.wikipedia.org/wiki/Hierarchical_clustering
    """

    def __init__(
        self,
        data_or_processor: Union[np.ndarray, DistanceMatrix],
        labels: Optional[List[str]] = None,
        linkage: str = "average",
        ordering_method: str = "recursive",
        max_merges: Optional[int] = None,
    ):
        if isinstance(data_or_processor, DistanceMatrix):
            self.dist_processor = data_or_processor
        elif isinstance(data_or_processor, np.ndarray):
            self.dist_processor = DistanceMatrix(data_or_processor, metric="euclidean")
        else:
            raise TypeError("`data_or_processor` must be a numpy array or a DistanceMatrix instance.")

        self.n_samples = self.dist_processor.n_samples
        self.distance_matrix = self.dist_processor.get_distance_matrix()
        self.condensed_distance_matrix = self.dist_processor.get_condensed_matrix()

        valid_linkages = ["single", "complete", "average"]
        if linkage not in valid_linkages:
            raise ValueError(f"Unsupported linkage method: {linkage}. Choose from: {valid_linkages}")
        self.linkage_method = linkage

        self.labels = labels if labels is not None else [str(i) for i in range(self.n_samples)]
        if len(self.labels) != self.n_samples:
            raise ValueError("Number of labels must match number of data samples.")

        max_possible_merges = self.n_samples - 1 if self.n_samples > 1 else 0
        self.num_merges_to_perform = max_merges if max_merges is not None else max_possible_merges
        if not (0 <= self.num_merges_to_perform <= max_possible_merges):
            raise ValueError(f"max_merges must be between 0 and {max_possible_merges}.")

        self.cluster_map: Dict[int, Dict[str, Any]] = {}
        self.merge_history: List[Dict[str, Any]] = []
        self.linkage_matrix_ = np.zeros((self.num_merges_to_perform, 4))
        self.leaf_order: List[int] = list(range(self.n_samples))
        self._cluster_id_counter: int = self.n_samples
        self.children_ = np.array([]).reshape(0, 2)
        self.distances_ = np.array([])
        self.labels_ = np.zeros(self.n_samples, dtype=int)

    def fit(self) -> "HierarchicalClustering":
        """
        Fits the hierarchical clustering model to the data.
        """
        if self.n_samples <= 1:
            return self

        # Use scipy's highly optimized linkage function
        self.linkage_matrix_ = scipy_linkage(self.condensed_distance_matrix, method=self.linkage_method)

        if self.num_merges_to_perform < self.n_samples - 1:
            self.linkage_matrix_ = self.linkage_matrix_[:self.num_merges_to_perform]

        self.children_ = self.linkage_matrix_[:, :2].astype(int)
        self.distances_ = self.linkage_matrix_[:, 2]
        
        # Create a simple labeling based on the final clusters
        self._assign_labels()
        
        return self

    def _assign_labels(self):
        """Assigns cluster labels to each sample."""
        n_clusters = self.n_samples - len(self.linkage_matrix_)
        self.labels_ = np.arange(self.n_samples)
        
        cluster_map = {i: i for i in range(self.n_samples)}
        
        for i, merge in enumerate(self.linkage_matrix_[:, :2].astype(int)):
            cluster_id1, cluster_id2 = merge
            new_cluster_id = self.n_samples + i
            
            # Find all points belonging to the merged clusters
            points1 = [p for p, c in cluster_map.items() if c == cluster_id1]
            points2 = [p for p, c in cluster_map.items() if c == cluster_id2]
            
            for p in points1 + points2:
                cluster_map[p] = new_cluster_id

        # Final assignment
        final_clusters = np.unique(list(cluster_map.values()))
        label_map = {cid: i for i, cid in enumerate(final_clusters)}
        
        for i in range(self.n_samples):
            self.labels_[i] = label_map[cluster_map[i]]

    def _get_summary_data(self):
        """Returns a dictionary containing summary data for the model."""
        summary = self._create_general_summary(
            self.n_samples,
            self.n_samples, # n_objects should be n_samples. n_variables is not used in this function.
            No_Clusters=f"{self.n_samples - len(self.linkage_matrix_)}",
            Linkage=self.linkage_method
        )
        return summary