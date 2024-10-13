import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from chemtools.utility import heatmap
from chemtools.utility import annotate_heatmap


def plot_correlation_matrix(pca_object, cmap="coolwarm", threshold=None):
    """Plots the correlation matrix of the data used in the PCA.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        cmap (str, optional): The colormap for the heatmap. Defaults to "coolwarm".
        threshold (float, optional): The threshold for displaying text. If None,
            the midpoint of the colormap is used. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(pca_object.correlation_matrix, cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, cmap=cmap, label="Correlation value")
    ax.set_xticks(np.arange(len(pca_object.variables)))
    ax.set_yticks(np.arange(len(pca_object.variables)))
    ax.set_xticklabels(pca_object.variables)
    ax.set_yticklabels(pca_object.variables)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Calculate the midpoint of the colormap if threshold is None
    if threshold is None:
        norm = plt.Normalize(
            vmin=pca_object.correlation_matrix.min(),
            vmax=pca_object.correlation_matrix.max(),
        )
        midpoint = (
            pca_object.correlation_matrix.max() + pca_object.correlation_matrix.min()
        ) / 2
        threshold = norm(midpoint)

    # Add correlation values as text
    for i in range(len(pca_object.variables)):
        for j in range(len(pca_object.variables)):
            color = (
                "black"
                if abs(pca_object.correlation_matrix[i, j]) < threshold
                else "white"
            )
            text = ax.text(
                j,
                i,
                f"{pca_object.correlation_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
            )

    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_eigenvalues_greater_than_one(pca_object, ax=None):
    """Plots the eigenvalues and highlights those greater than one.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None,
                                            a new figure and axes will be created.
                                            Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(pca_object.V_ordered, marker="o", linestyle="-", color="b")
        ax.set_ylabel("Eigenvalue")
        ax.legend(loc="best")
        ax.set_title("Eigenvalues with Threshold")
        ax.set_xlabel(r"$PC_i$")
        ax.grid(True)  # Add a grid for better visualization
    else:
        fig = ax.get_figure()

    num_eigenvalues_greater_than_one = np.argmax(pca_object.V_ordered < 1)
    ax.axvline(
        x=num_eigenvalues_greater_than_one - 0.5,
        color="brown",
        linestyle="-",
        label="Eigenvalues > 1",  # Clearer label
    )
    ax.set_xticks(range(len(pca_object.PC_index)))
    ax.set_xticklabels(pca_object.PC_index)

    return fig, ax


def plot_eigenvalues_variance(pca_object, ax=None):
    """Plots the percentage of variance explained by each principal component.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None,
                                            a new figure and axes will be created.
                                            Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(r"$PC_i$")
        ax.set_ylabel("Variance Explained (%)")
        ax.legend(loc="best")
        ax.set_title("Variance Explained by Each Principal Component")
        ax.grid(True)
    else:
        fig = ax.get_figure()

    variance_explained = (pca_object.V_ordered / pca_object.V_ordered.sum()) * 100

    ax.bar(
        x=pca_object.PC_index,
        height=variance_explained,
        fill=False,
        edgecolor="darkorange",
        label="Variance Explained (%)",
    )
    ax.set_xticks(range(len(pca_object.PC_index)))
    ax.set_xticklabels(pca_object.PC_index)

    return fig, ax


def plot_cumulative_variance(pca_object, ax=None):
    """Plots the cumulative percentage of variance explained by the principal components.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None,
                                            a new figure and axes will be created.
                                            Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(r"$PC_i$")
        ax.set_ylabel("Cumulative Variance Explained (%)")  # More accurate label
        ax.legend(loc="best")
        ax.set_title(
            "Cumulative Variance Explained by Principal Components"
        )  # Clearer title
        ax.grid(True)
    else:
        fig = ax.get_figure()

    cumulative_variance = (
        np.cumsum(pca_object.V_ordered / pca_object.V_ordered.sum()) * 100
    )

    ax.bar(
        x=pca_object.PC_index,
        height=cumulative_variance,
        fill=False,
        edgecolor="black",
        linestyle="--",
        width=0.6,
        label="Cumulative Variance Explained (%)",  # Improved label
    )
    ax.set_xticks(range(len(pca_object.PC_index)))
    ax.set_xticklabels(pca_object.PC_index)

    return fig, ax


def plot_average_eigenvalue_criterion(pca_object, ax=None):
    """Plots the eigenvalues and marks the Average Eigenvalue Criterion (AEC).

    The AEC suggests retaining components with eigenvalues greater than
    the average eigenvalue.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None,
                                            a new figure and axes will be created.
                                            Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(pca_object.V_ordered, marker="o", linestyle="-", color="b")
        ax.set_ylabel("Eigenvalue")
        ax.set_xlabel(r"$PC_i$")
        ax.legend(loc="best")
        ax.set_title("Eigenvalues with Average Eigenvalue Criterion")
        ax.grid(True)
    else:
        fig = ax.get_figure()


    ax.axvline(
        x=np.argmax(pca_object.V_ordered < pca_object.V_ordered.mean()) - 0.5,
        color="red",
        alpha=0.5,
        linestyle="-",
        label="AEC (Average Eigenvalue)",
    )
    ax.set_xticks(range(len(pca_object.PC_index)))
    ax.set_xticklabels(pca_object.PC_index)
    return fig, ax


def plot_KP_criterion(pca_object, ax=None):
    """Plots the eigenvalues and indicates the Kaiser-Piggott (KP) criterion.

    The KP criterion suggests retaining components with eigenvalues greater
    than a specific threshold, often set to 1.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None,
                                            a new figure and axes will be created.
                                            Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(pca_object.V_ordered, marker="o", linestyle="-", color="b")
        ax.set_xlabel(r"$PC_i$")
        ax.set_ylabel("Eigenvalue")
        ax.legend(loc="best")
        ax.set_title("Eigenvalues with Kaiser-Piggott Criterion")
        ax.grid(True)
    else:
        fig = ax.get_figure()

    rank = np.linalg.matrix_rank(pca_object.correlation_matrix)
    sum_term = sum(
        pca_object.V[m] / pca_object.V.sum() - 1 / pca_object.V.size
        for m in range(rank)
    )
    x = (
        round(
            1
            + (pca_object.V.size - 1)
            * (
                1
                - (
                    (sum_term + (pca_object.V.size - rank) ** (1 / pca_object.V.size))
                    / (2 * (pca_object.V.size - 1) / pca_object.V.size)
                )
            )
        )
        - 1
    )

    ax.axvline(
        x=x,
        color="purple",
        alpha=0.5,
        linestyle="--",
        label="KP Criterion",  # More descriptive label
    )
    ax.set_xticks(range(len(pca_object.PC_index)))
    ax.set_xticklabels(pca_object.PC_index)

    return fig, ax


def plot_KL_criterion(pca_object, ax=None):
    """Plots the eigenvalues and marks the KL criterion for component selection.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None,
                                            a new figure and axes will be created.
                                            Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(pca_object.V_ordered, marker="o", linestyle="-", color="b")
        ax.set_xlabel(r"$PC_i$")
        ax.set_ylabel("Eigenvalue")
        ax.legend(loc="best")
        ax.set_title("Eigenvalues with KL Criterion")
        ax.grid(True)
    else:
        fig = ax.get_figure()

    rank = np.linalg.matrix_rank(pca_object.correlation_matrix)
    sum_term = sum(
        pca_object.V[m] / pca_object.V.sum() - 1 / pca_object.V.size
        for m in range(rank)
    )
    x = (
        round(
            pca_object.V.size
            ** (
                1
                - (sum_term + (pca_object.V.size - rank) ** (1 / pca_object.V.size))
                / (2 * (pca_object.V.size - 1) / pca_object.V.size)
            )
        )
        - 1
    )

    ax.axvline(
        x=x,
        color="cyan",
        alpha=0.5,
        linestyle="-",
        label="KL Criterion",
    )
    ax.set_xticks(range(len(pca_object.PC_index)))
    ax.set_xticklabels(pca_object.PC_index)

    return fig, ax


def plot_CAEC_criterion(pca_object, ax=None):
    """Plots the eigenvalues and marks the CAEC (Cumulative Average
    Eigenvalue Criterion).

    The CAEC suggests retaining components with eigenvalues greater than
    a percentage (typically 70%) of the average eigenvalue.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None,
                                            a new figure and axes will be created.
                                            Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(pca_object.V_ordered, marker="o", linestyle="-", color="b")
        ax.set_xlabel(r"$PC_i$")
        ax.set_ylabel("Eigenvalue")
        ax.legend(loc="best")
        ax.set_title("Eigenvalues with Cumulative Average Eigenvalue Criterion")
        ax.grid(True)
    else:
        fig = ax.get_figure()

    ax.axvline(
        x=np.argmax(pca_object.V_ordered < 0.7 * pca_object.V_ordered.mean()) - 0.5,
        color="blue",
        alpha=0.5,
        linestyle="--",
        label="CAEC (70% of Avg. Eigenvalue)",
    )
    ax.set_xticks(range(len(pca_object.PC_index)))
    ax.set_xticklabels(pca_object.PC_index)

    return fig, ax


def plot_broken_stick(pca_object, ax=None):
    """Plots the broken stick criterion for selecting the number of PCs.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None,
                                            a new figure and axes will be created.
                                            Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(r"$PC_i$")
        ax.set_ylabel("Percentage of Total Variance (%)")
        ax.legend(loc="best")
        ax.set_title("Broken Stick Criterion")
        ax.grid(True)
    else:
        fig = ax.get_figure()
    n = pca_object.V_ordered.shape[0]
    dm = (100 / n) * np.cumsum(1 / np.arange(1, n + 1)[::-1])

    ax.plot(pca_object.PC_index, dm, color="lightgreen", label="Broken Stick")
    ax.set_xticks(range(len(pca_object.PC_index)))
    ax.set_xticklabels(pca_object.PC_index)

    return fig, ax


def plot_eigenvalue(pca_object):
    """Plots various eigenvalue criteria for component selection."""
    fig, ax1 = plt.subplots(figsize=(15, 10))

    # Plot all criteria on the same plot with different styles
    plot_eigenvalues_greater_than_one(pca_object, ax=ax1)
    plot_eigenvalues_variance(pca_object, ax=ax1)
    plot_cumulative_variance(pca_object, ax=ax1)
    plot_average_eigenvalue_criterion(pca_object, ax=ax1)
    plot_KP_criterion(pca_object, ax=ax1)
    plot_KL_criterion(pca_object, ax=ax1)
    plot_CAEC_criterion(pca_object, ax=ax1)
    plot_broken_stick(pca_object, ax=ax1)
    ax1.set_xlabel(r"$PC_i$")
    ax1.set_ylabel("Percentage Variance (%)")
    ax1.legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_hotteling_t2_vs_q(pca_object, show=True):
    """Plots the Hotelling's T2 statistic versus the Q statistic (Squared Prediction Error).

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        show (bool, optional): Whether to display the plot immediately.
                               Defaults to True.
    """

    plt.figure(figsize=(8, 6))
    for i in range(len(pca_object.Q)):
        plt.plot(
            pca_object.Q[i],
            pca_object.T2[i],
            "o",
            label=pca_object.objects[i],
        )

    plt.xlabel(r"$Q$ (Squared Prediction Error)")
    plt.ylabel(r"$Hotelling's T^2$")
    plt.legend(loc="best")
    plt.title("Hotelling's T2 vs. Q")
    plt.grid(True)

    if show:
        plt.tight_layout()
        plt.show()


def plot_pci_contribution(pca_object, text_color="black", show=True):
    """Plots the contribution of each variable to each principal component.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        text_color (str, optional): The color of the text labels.
                                    Defaults to "black".
        show (bool, optional): Whether to display the plot immediately.
                               Defaults to True.
    """

    plt.figure(figsize=(10, 6))
    for i in range(pca_object.W.shape[1]):
        plt.plot(
            np.arange(pca_object.n_variables),
            pca_object.W[:, i],
            marker="o",
            markerfacecolor="none",
            label=f"PC$_{i+1}$",
        )
    plt.title("Contributions of Variables to Each PC")
    plt.xticks(
        np.arange(pca_object.n_variables),
        pca_object.variables,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    plt.legend(labelcolor=text_color)
    plt.xlabel("Variable")
    plt.ylabel("Value of Loading")
    plt.grid(True)

    if show:
        plt.tight_layout()
        plt.show()


def plot_loadings(
    pca_object, arrows=True, text_color="black", components=None, show=True
):
    """Plots the loadings of the principal components.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        arrows (bool, optional): Whether to display arrows for the loadings.
                                 Defaults to True.
        text_color (str, optional): The color of the text labels.
                                    Defaults to "black".
        components (tuple, optional): The components to plot.
                                     Defaults to None (all components).
        show (bool, optional): Whether to display the plot immediately.
                               Defaults to True.
    """
    if components is not None:
        i, j = components
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.suptitle(
            f"Loadings plot PC{i+1} vs PC{j+1}",
            fontsize=24,
            y=1,
            color=text_color,
        )

        x_data = pca_object.W[:, i]
        y_data = pca_object.W[:, j]
        colors = pca_object.variables_colors
        ax.scatter(x_data, y_data, c=colors, label=pca_object.variables)

        if arrows:
            for d in range(pca_object.n_variables):
                ax.arrow(
                    0,
                    0,
                    pca_object.W[d, i],
                    pca_object.W[d, j],
                    length_includes_head=True,
                    width=0.01,
                    color=text_color,
                    alpha=0.3,
                )

        for d in range(pca_object.n_variables):
            position = (
                ["left", "bottom"]
                if pca_object.W[d, i] > pca_object.W[:, i].mean()
                else ["right", "top"]
            )
            ax.annotate(
                text=pca_object.variables[d],
                xy=(pca_object.W[d, i], pca_object.W[d, j]),
                ha=position[0],
                va=position[1],
                color=text_color,
            )

        ax.set_xlabel(rf"PC$_{i+1}$")
        ax.set_ylabel(rf"PC$_{j+1}$")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=text_color,
        )
        plt.tight_layout()

    else:
        height_ratios = [1] * pca_object.n_component
        width_ratios = [1] * pca_object.n_component

        fig, axs = plt.subplots(
            pca_object.n_component,
            pca_object.n_component,
            figsize=(5 * pca_object.n_component, 5 * pca_object.n_component),
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
        )
        fig.suptitle("Loadings plot", fontsize=24, y=1, color=text_color)

        for i in range(pca_object.n_component):
            for j in range(pca_object.n_component):
                ax = axs[i, j] if pca_object.n_component > 1 else axs
                if i != j:

                    x_data = pca_object.W[:, i]
                    y_data = pca_object.W[:, j]
                    colors = pca_object.variables_colors
                    ax.scatter(x_data, y_data, c=colors, label=pca_object.variables)

                    if arrows:
                        for d in range(pca_object.n_variables):
                            ax.arrow(
                                0,
                                0,
                                pca_object.W[d, i],
                                pca_object.W[d, j],
                                length_includes_head=True,
                                width=0.01,
                                color=text_color,
                                alpha=0.3,
                            )

                    for d in range(pca_object.n_variables):
                        position = (
                            ["left", "bottom"]
                            if pca_object.W[d, i] > pca_object.W[:, i].mean()
                            else ["right", "top"]
                        )
                        ax.annotate(
                            text=pca_object.variables[d],
                            xy=(pca_object.W[d, i], pca_object.W[d, j]),
                            ha=position[0],
                            va=position[1],
                            color=text_color,
                        )
                    ax.set_xlabel(rf"PC$_{i+1}$")
                    ax.set_ylabel(rf"PC$_{j+1}$")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        rf"PC$_{i+1}$",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=20,
                        color=text_color,
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.add_patch(
                        patches.Rectangle(
                            (0, 0),
                            1,
                            1,
                            fill=False,
                            transform=ax.transAxes,
                            clip_on=False,
                        )
                    )

        handles, labels = ax.get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=text_color,
        )
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

    if show:
        plt.show()


def plot_scores(
    pca_object, label_point=False, text_color="black", components=None, show=True
):
    """Plots the scores of the principal components.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        label_point (bool, optional): Whether to label the points.
                                      Defaults to False.
        text_color (str, optional): The color of the text labels.
                                    Defaults to "black".
        components (tuple, optional): The components to plot.
                                     Defaults to None (all components).
        show (bool, optional): Whether to display the plot immediately.
                               Defaults to True.
    """
    if components is not None:
        i, j = components
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.suptitle(
            f"Scores plot PC{i+1} vs PC{j+1}",
            fontsize=24,
            y=1,
            color=text_color,
        )

        x_data = pca_object.T[:, i]
        y_data = pca_object.T[:, j]
        colors = pca_object.objects_colors
        ax.scatter(x_data, y_data, c=colors, label=pca_object.objects)

        if label_point:
            for d in range(pca_object.n_objects):
                ax.annotate(
                    text=pca_object.objects[d][0],
                    xy=(pca_object.T[d, i], pca_object.T[d, j]),
                    color=text_color,
                )

        ax.set_xlabel(f"PC{i+1}")
        ax.set_ylabel(f"PC{j+1}")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=text_color,
        )
        plt.tight_layout()

    else:
        height_ratios = [1] * pca_object.n_component
        width_ratios = [1] * pca_object.n_component

        fig, axs = plt.subplots(
            pca_object.n_component,
            pca_object.n_component,
            figsize=(5 * pca_object.n_component, 5 * pca_object.n_component),
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
        )
        fig.suptitle("Scores plot", fontsize=24, y=1, color=text_color)

        for i in range(pca_object.n_component):
            for j in range(pca_object.n_component):
                ax = axs[i, j] if pca_object.n_component > 1 else axs
                if i != j:
                    x_data = pca_object.T[:, i]
                    y_data = pca_object.T[:, j]
                    colors = pca_object.objects_colors
                    ax.scatter(x_data, y_data, c=colors, label=pca_object.objects)

                    if label_point:
                        for d in range(pca_object.n_objects):
                            ax.annotate(
                                text=pca_object.objects[d][0],
                                xy=(pca_object.T[d, i], pca_object.T[d, j]),
                                color=text_color,
                            )
                    ax.set_xlabel(f"PC{i+1}")
                    ax.set_ylabel(f"PC{j+1}")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"PC{i+1}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=20,
                        color=text_color,
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.add_patch(
                        patches.Rectangle(
                            (0, 0),
                            1,
                            1,
                            fill=False,
                            transform=ax.transAxes,
                            clip_on=False,
                        )
                    )

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=text_color,
        )
        plt.tight_layout()

    if show:
        plt.show()


def plot_biplot(
    pca_object,
    label_point=False,
    subplot_dimensions=[5, 5],
    text_color="black",
    components=None,
    show_legend=True,
    show=True,
):
    """Plots a biplot, combining scores and loadings in one plot.

    Args:
        pca_object (PrincipalComponentAnalysis): The fitted PCA object.
        label_point (bool, optional): Whether to label the score points.
                                      Defaults to False.
        subplot_dimensions (list, optional): Dimensions of subplots.
                                             Defaults to [5, 5].
        text_color (str, optional): The color for text labels.
                                    Defaults to "black".
        components (tuple, optional):  The principal components to display.
                                      Defaults to None (all components).
        show_legend (bool, optional): Whether to display the legend.
                                     Defaults to True.
        show (bool, optional): Whether to display the plot immediately.
                               Defaults to True.
    """
    if components is not None:
        i, j = components
        fig, ax = plt.subplots(figsize=(subplot_dimensions[0], subplot_dimensions[1]))
        fig.suptitle(f"Biplot PC{i+1} vs PC{j+1}", fontsize=24, y=1, color=text_color)

        x_data = pca_object.T[:, i]
        y_data = pca_object.T[:, j]
        colors = pca_object.objects_colors
        scatter = ax.scatter(x_data, y_data, c=colors, label=pca_object.objects)

        if label_point:
            for e in range(pca_object.n_objects):
                ax.annotate(
                    text=pca_object.objects[e],
                    xy=(pca_object.T[e, i], pca_object.T[e, j]),
                    color=text_color,
                )

        for d in range(pca_object.n_variables):
            position = (
                ["left", "bottom"]
                if pca_object.W[d, i] > pca_object.W[:, i].mean()
                else ["right", "top"]
            )
            ax.arrow(
                0,
                0,
                pca_object.W[d, i],
                pca_object.W[d, j],
                length_includes_head=True,
                width=0.01,
                color=text_color,
                alpha=0.3,
            )
            ax.annotate(
                text=pca_object.variables[d],
                xy=(pca_object.W[d, i], pca_object.W[d, j]),
                ha=position[0],
                va=position[1],
                color=text_color,
            )
        ax.set_xlabel(f"PC{i+1}")
        ax.set_ylabel(f"PC{j+1}")

        if show_legend:
            handles, labels = scatter.legend_elements()
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                labelcolor=text_color,
            )
        plt.tight_layout()

    else:
        height_ratios = [1] * pca_object.n_component
        width_ratios = [1] * pca_object.n_component

        fig, axs = plt.subplots(
            pca_object.n_component,
            pca_object.n_component,
            figsize=(
                subplot_dimensions[0] * pca_object.n_component,
                subplot_dimensions[1] * pca_object.n_component,
            ),
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
        )
        fig.suptitle("Biplots plot", fontsize=24, y=1, color=text_color)

        all_handles = []
        all_labels = []

        for i in range(pca_object.n_component):
            for j in range(pca_object.n_component):
                ax = axs[i, j] if pca_object.n_component > 1 else axs
                if i != j:

                    x_data = pca_object.T[:, i]
                    y_data = pca_object.T[:, j]
                    colors = pca_object.objects_colors
                    scatter = ax.scatter(
                        x_data, y_data, c=colors, label=pca_object.objects
                    )

                    if label_point:
                        for e in range(pca_object.n_objects):
                            ax.annotate(
                                text=pca_object.objects[e],
                                xy=(pca_object.T[e, i], pca_object.T[e, j]),
                                color=text_color,
                            )

                    for d in range(pca_object.n_variables):
                        position = (
                            ["left", "bottom"]
                            if pca_object.W[d, i] > pca_object.W[:, i].mean()
                            else ["right", "top"]
                        )
                        ax.arrow(
                            0,
                            0,
                            pca_object.W[d, i],
                            pca_object.W[d, j],
                            length_includes_head=True,
                            width=0.01,
                            color=text_color,
                            alpha=0.3,
                        )
                        ax.annotate(
                            text=pca_object.variables[d],
                            xy=(
                                pca_object.W[d, i],
                                pca_object.W[d, j],
                            ),
                            ha=position[0],
                            va=position[1],
                            color=text_color,
                        )
                    ax.set_xlabel(f"PC{i+1}")
                    ax.set_ylabel(f"PC{j+1}")

                    handles, labels = scatter.legend_elements()
                    all_handles.extend(handles)
                    all_labels.extend(labels)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"PC{i+1}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=20,
                        color=text_color,
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.add_patch(
                        patches.Rectangle(
                            (0, 0),
                            1,
                            1,
                            fill=False,
                            transform=ax.transAxes,
                            clip_on=False,
                        )
                    )

        if show_legend:
            fig.legend(
                all_handles,
                all_labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
        plt.tight_layout()
        plt.show()
