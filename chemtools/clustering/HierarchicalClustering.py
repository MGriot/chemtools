# Necessary imports (ensure all are present)
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import optimal_leaf_ordering

# from scipy.spatial.distance import pdist, squareform # No longer needed directly here
from scipy.cluster.hierarchy import linkage as scipy_linkage
import sys
from typing import List, Optional, Dict, Tuple, Union, Set, Any
import heapq
from chemtools.preprocessing.distance_matrix import DistanceMatrix

# Assume DistanceMatrix class from above is defined here


class HierarchicalClustering:
    """
    Performs Agglomerative Hierarchical Clustering using a precomputed
    distance matrix or by calculating it from data. Allows limiting
    the number of merge steps.

    Attributes:
        n_samples (int): Number of data points.
        linkage_method (str): Linkage criterion ('single', 'complete', 'average').
        ordering_method (str): Leaf ordering method ('recursive', 'original', 'olo', 'tsp').
        labels (List[str]): Labels for data points.
        distance_matrix (np.ndarray): Full square distance matrix used.
        condensed_distance_matrix (np.ndarray): Condensed distance matrix used.
        linkage_matrix_ (np.ndarray): SciPy-compatible linkage matrix (Z), potentially truncated.
        leaf_order (List[int]): Final order of leaves for plotting.
        cluster_map (Dict[int, Dict[str, Any]]): Stores all clusters created.
        num_merges_to_perform (int): Actual number of merges that will be done.
        merge_history: List[Dict[str, Any]]: History of merges performed.
        children_ (np.ndarray): Cluster merge children (for sklearn compatibility).
        distances_ (np.ndarray): Distances at which merges occurred.
        labels_ (np.ndarray): Cluster labels for each sample.

    Returns:
        HierarchicalClustering: Returns self after fitting.
            - linkage_matrix_: The hierarchical clustering encoded as a linkage matrix.
            - children_: A list of length n-1 of clusters merged at each step.
            - distances_: The distances between merged clusters at each step.
            - labels_: Cluster labels for each input sample.
            - leaf_order: The order of leaves in the dendrogram for visualization.
            - cluster_map: Dictionary containing all clusters and their properties.
    """

    def __init__(
        self,
        data_or_processor: Union[np.ndarray, DistanceMatrix],
        labels: Optional[List[str]] = None,
        linkage: str = "average",
        ordering_method: str = "recursive",
        max_merges: Optional[int] = None,
    ):
        # --- 1. Process Input: Data or Processor ---
        if isinstance(data_or_processor, DistanceMatrix):
            self.dist_processor = data_or_processor
            self.n_samples = self.dist_processor.n_samples
            print("Using provided DistanceMatrix.")
        elif isinstance(data_or_processor, np.ndarray):
            print("Input is data array. Creating DistanceMatrix...")
            self.dist_processor = DistanceMatrix(data_or_processor, metric="euclidean")
            self.n_samples = self.dist_processor.n_samples
        else:
            raise TypeError(
                "`data_or_processor` must be a numpy array or a DistanceMatrix instance."
            )

        self.distance_matrix = self.dist_processor.get_distance_matrix()
        self.condensed_distance_matrix = self.dist_processor.get_condensed_matrix()

        # --- 2. Validate Parameters ---
        valid_linkages = ["single", "complete", "average"]
        if linkage not in valid_linkages:
            raise ValueError(
                f"Unsupported linkage method: {linkage}. Choose from: {', '.join(valid_linkages)}"
            )
        self.linkage_method = linkage

        valid_ordering = ["original", "recursive", "olo", "tsp"]
        if ordering_method not in valid_ordering:
            raise ValueError(
                f"Unsupported ordering method: {ordering_method}. Choose from: {', '.join(valid_ordering)}"
            )
        self.ordering_method = ordering_method

        self.labels = (
            labels if labels is not None else [str(i) for i in range(self.n_samples)]
        )
        if len(self.labels) != self.n_samples:
            raise ValueError("Number of labels must match number of data samples.")

        # --- 3. Determine Number of Merges ---
        max_possible_merges = self.n_samples - 1
        if max_possible_merges < 1 and self.n_samples > 0:
            max_possible_merges = 0
        if max_merges is None:
            self.num_merges_to_perform = max_possible_merges
        else:
            if not (1 <= max_merges <= max_possible_merges):
                raise ValueError(
                    f"max_merges ({max_merges}) must be between 1 and "
                    f"n_samples - 1 ({max_possible_merges})."
                )
            self.num_merges_to_perform = max_merges
            print(
                f"INFO: Will perform a maximum of {self.num_merges_to_perform} merge steps."
            )
            if (
                self.ordering_method in ["olo", "tsp"]
                and self.num_merges_to_perform < max_possible_merges
            ):
                print(
                    f"Warning: Ordering method '{self.ordering_method}' relies on the full hierarchy. "
                    "Results might be suboptimal or based on partial structure when max_merges is set."
                )

        # --- 4. Initialize Clustering State ---
        self.cluster_map: Dict[int, Dict[str, Any]] = {}
        self.merge_history: List[Dict[str, Any]] = []
        self.linkage_matrix_ = np.zeros((self.num_merges_to_perform, 4))
        self.leaf_order: List[int] = list(range(self.n_samples))
        self._cluster_id_counter: int = self.n_samples
        self.leaf_clusters: List[Dict[str, Any]] = []

        # Initialize sklearn-compatible attributes for later
        self.children_ = np.array([]).reshape(0, 2)
        self.distances_ = np.array([])
        self.labels_ = np.zeros(self.n_samples, dtype=int)

    def _get_cluster_color(self, index: int) -> Tuple[float, float, float]:
        colors = plt.cm.tab10.colors
        return colors[index % len(colors)]

    def _cluster_distance(self, c1_id: int, c2_id: int) -> float:
        c1 = self.cluster_map.get(c1_id)
        c2 = self.cluster_map.get(c2_id)
        if not c1 or not c2:
            print(f"Warning: Cluster ID {c1_id if not c1 else c2_id} not found.")
            return np.inf
        points1 = c1["points"]
        points2 = c2["points"]
        if not points1 or not points2:
            return 0.0
        try:
            dists = self.distance_matrix[np.ix_(points1, points2)]
        except IndexError:
            print(
                f"Warning: IndexError accessing distance matrix for points {points1} vs {points2}."
            )
            return np.inf
        if dists.size == 0:
            return 0.0
        if self.linkage_method == "single":
            return np.min(dists)
        elif self.linkage_method == "complete":
            return np.max(dists)
        elif self.linkage_method == "average":
            return np.mean(dists)
        else:
            raise ValueError(
                f"Internal Error: Unsupported linkage method '{self.linkage_method}'."
            )

    def _merge_clusters(
        self, c1_id: int, c2_id: int, merge_dist: float
    ) -> Dict[str, Any]:
        c1 = self.cluster_map[c1_id]
        c2 = self.cluster_map[c2_id]
        new_id = self._cluster_id_counter
        self._cluster_id_counter += 1
        merged_points = c1["points"] + c2["points"]
        new_size = c1["size"] + c2["size"]
        order1 = c1.get("order", [])
        order2 = c2.get("order", [])
        merged_order = order1 + order2
        if order1 and order2:
            try:
                dist_forward = self.distance_matrix[order1[-1], order2[0]]
                dist_reverse_c2 = self.distance_matrix[order1[-1], order2[-1]]
                if dist_reverse_c2 < dist_forward:
                    merged_order = order1 + order2[::-1]
            except IndexError:
                print(
                    f"Warning (_merge_clusters): IndexError accessing distance matrix for heuristic ordering. Orders: {order1}, {order2}. Using default."
                )
            except Exception as e:
                print(
                    f"Warning (_merge_clusters): Error during heuristic ordering: {e}. Using default."
                )
        c1_x, c2_x = c1.get("x", 0.0), c2.get("x", 0.0)
        c1_size, c2_size = c1.get("size", 1), c2.get("size", 1)
        internal_x = (
            (c1_x * c1_size + c2_x * c2_size) / new_size
            if new_size > 0
            else (c1_x + c2_x) / 2.0
        )
        new_cluster = {
            "id": new_id,
            "points": merged_points,
            "x": internal_x,
            "height": float(merge_dist),
            "size": new_size,
            "left_id": c1_id,
            "right_id": c2_id,
            "left_cluster": c1,
            "right_cluster": c2,
            "color": c1.get("color") if c1_size >= c2_size else c2.get("color"),
            "order": merged_order,
        }
        return new_cluster

    def fit(self) -> "HierarchicalClustering":
        self.cluster_map = {}
        active_cluster_ids: Set[int] = set(range(self.n_samples))
        self.leaf_clusters = []

        for i in range(self.n_samples):
            cluster = {
                "id": i,
                "points": [i],
                "x": float(i),
                "height": 0.0,
                "size": 1,
                "color": self._get_cluster_color(i),
                "order": [i],
                "left_id": -1,
                "right_id": -1,
            }
            self.cluster_map[i] = cluster
            self.leaf_clusters.append(cluster)

        self.merge_history = []
        if self.n_samples <= 1:
            print("INFO: Only 0 or 1 sample, no merges to perform.")
            self.num_merges_to_perform = 0
            self.linkage_matrix_ = np.zeros((0, 4))

        for merge_step in range(self.num_merges_to_perform):
            if len(active_cluster_ids) <= 1:
                print(
                    f"Warning: Only {len(active_cluster_ids)} cluster(s) remain at step {merge_step+1}. "
                    f"Stopping merge early (expected {self.num_merges_to_perform} steps)."
                )
                self.linkage_matrix_ = self.linkage_matrix_[:merge_step, :]
                self.num_merges_to_perform = merge_step
                break

            min_dist = np.inf
            best_pair: Optional[Tuple[int, int]] = None
            active_list = list(active_cluster_ids)
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    id1, id2 = active_list[i], active_list[j]
                    dist = self._cluster_distance(id1, id2)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (id1, id2)

            if best_pair is None:
                print(
                    f"Error: Could not find a pair to merge at step {merge_step+1}. Stopping."
                )
                self.linkage_matrix_ = self.linkage_matrix_[:merge_step, :]
                self.num_merges_to_perform = merge_step
                break

            c1_id, c2_id = best_pair
            new_cluster = self._merge_clusters(c1_id, c2_id, float(min_dist))
            new_id = new_cluster["id"]

            merged_id1, merged_id2 = sorted(best_pair)
            self.linkage_matrix_[merge_step, 0] = float(merged_id1)
            self.linkage_matrix_[merge_step, 1] = float(merged_id2)
            self.linkage_matrix_[merge_step, 2] = float(min_dist)
            self.linkage_matrix_[merge_step, 3] = float(new_cluster["size"])

            self.merge_history.append(new_cluster)
            self.cluster_map[new_id] = new_cluster
            active_cluster_ids.remove(c1_id)
            active_cluster_ids.remove(c2_id)
            active_cluster_ids.add(new_id)

        print(
            f"Completed {self.num_merges_to_perform} merge steps. "
            f"{len(active_cluster_ids)} clusters remaining."
        )

        self._determine_leaf_order()
        self._assign_final_x_coordinates()

        # --- Set sklearn-compatible attributes ---
        if self.linkage_matrix_.shape[0] > 0:
            self.children_ = self.linkage_matrix_[:, :2].astype(int)
            self.distances_ = self.linkage_matrix_[:, 2]
        else:
            self.children_ = np.array([]).reshape(0, 2)
            self.distances_ = np.array([])

        # Assign labels_ (simple labeling: each point assigned to cluster 0..k)
        # Here we assign labels based on the last merges - if partial merges, labels may be trivial
        self.labels_ = np.zeros(self.n_samples, dtype=int)
        # If full clustering done, assign all points to cluster 0 (single cluster)
        if self.num_merges_to_perform == self.n_samples - 1 and self.n_samples > 0:
            self.labels_ = np.zeros(self.n_samples, dtype=int)
        else:
            # Partial clustering: assign clusters based on remaining active clusters
            # Map each active cluster id to a label
            label_map = {}
            for label, cid in enumerate(active_cluster_ids):
                label_map[cid] = label
            for idx in range(self.n_samples):
                # Find cluster containing this point
                for cid in active_cluster_ids:
                    if idx in self.cluster_map[cid]["points"]:
                        self.labels_[idx] = label_map[cid]
                        break

        return self

    def _determine_leaf_order(self) -> None:
        final_order: List[int] = list(range(self.n_samples))
        max_possible_merges = self.n_samples - 1 if self.n_samples > 1 else 0
        is_partial_clustering = self.num_merges_to_perform < max_possible_merges

        if self.n_samples <= 1:
            self.leaf_order = final_order
            return

        try:
            if self.ordering_method == "original":
                final_order = list(range(self.n_samples))
            elif self.ordering_method == "recursive":
                if self.merge_history:
                    last_merged_node = self.merge_history[-1]
                    rec_order = last_merged_node.get("order")
                    if (
                        isinstance(rec_order, list)
                        and len(rec_order) == self.n_samples
                        and all(isinstance(i, int) for i in rec_order)
                        and set(rec_order) == set(range(self.n_samples))
                    ):
                        final_order = rec_order
            # OLO and TSP ordering could be implemented here if desired
        except Exception as e:
            print(f"Error in leaf ordering: {e}. Using default order.")

        self.leaf_order = final_order

    def _assign_final_x_coordinates(self) -> None:
        leaf_x = {leaf: float(i) for i, leaf in enumerate(self.leaf_order)}
        for cluster in self.cluster_map.values():
            if "points" in cluster:
                if len(cluster["points"]) == 1:
                    cluster["x"] = leaf_x[cluster["points"][0]]
                else:
                    cluster["x"] = np.mean([leaf_x[p] for p in cluster["points"]])
