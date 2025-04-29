import numpy as np
import matplotlib.pyplot as plt
def dendogram_plot(model, **kwargs):
    """
    Plot a dendrogram from a fitted HierarchicalClustering model.
    
    Parameters:
    -----------
    model : HierarchicalClustering
        Fitted hierarchical clustering model.
    **kwargs : dict
        Additional keyword arguments to pass to scipy.cluster.hierarchy.dendrogram.
    """
    from scipy.cluster.hierarchy import dendrogram

    if not hasattr(model, 'children_') or not hasattr(model, 'distances_'):
        raise ValueError("Model doesn't appear to be a fitted HierarchicalClustering instance.")

    # Calculate counts (as in the scikit-learn example)
    n_samples = len(getattr(model, 'labels_', []))
    if n_samples == 0:
        raise ValueError("Model doesn't have any labels.")

    counts = np.zeros(model.children_.shape[0])
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # Create linkage matrix for dendrogram
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the dendrogram
    return dendrogram(linkage_matrix, **kwargs)
