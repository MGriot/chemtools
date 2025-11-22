from ..base import BasePlotter
from chemtools.dimensional_reduction import FactorAnalysis
from chemtools.exploration import PrincipalComponentAnalysis
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches # For DimensionalityReductionPlot
from matplotlib.patches import Ellipse # For DimensionalityReductionPlot
from scipy.stats import chi2 # For DimensionalityReductionPlot's ellipse
from typing import List, Dict, Optional, Union, Tuple, Any # From Plotter, but good to have for type hints

# Assuming Plotter class is defined elsewhere and imported, e.g.:
# from .Plotter import Plotter
# Or if in the same file for testing:
# class Plotter: ... (full Plotter class definition)


# --- Subclass for Dimensionality Reduction Visualization ---
class DimensionalityReductionPlot(BasePlotter):
    """Class for dimensional reduction visualization."""

    def __init__(self, dim_reduction_model, library="matplotlib", theme="classic_professional_light", style_preset="default", **kwargs):
        super().__init__(library=library, theme=theme, style_preset=style_preset, **kwargs)
        self.dim_reduction_model = dim_reduction_model

        # Augment the self.colors dictionary from Plotter's theme
        self.colors["arrow_color"] = self.colors.get(
            "detail_medium_color", self.colors["text_color"]
        )
        self.colors["highlight_color"] = self.colors.get("accent_color", "#FF5733")
        self.colors["bar_edge_color"] = self.colors.get("theme_color", "#1f77b4")
        self.colors["ellipse_color"] = self.colors.get("accent_color", "#e76f51")

    def plot_correlation_matrix(self, cmap="coolwarm", **kwargs):
        if self.library != "matplotlib":
            print(
                f"plot_correlation_matrix is currently implemented for Matplotlib. Skipping for {self.library}."
            )
            return None

        processed_params = self._process_common_params(**kwargs)
        fig, ax = self._create_figure(figsize=processed_params["figsize"])

        if not hasattr(self.dim_reduction_model, "correlation_matrix"):
            raise AttributeError("Model must have 'correlation_matrix'.")

        im = ax.imshow(self.dim_reduction_model.correlation_matrix, cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors=self.colors["text_color"])
        cbar.outline.set_edgecolor(self.colors["text_color"])

        ax.tick_params(axis="x", colors=self.colors["text_color"])
        ax.tick_params(axis="y", colors=self.colors["text_color"])
        ax.grid(False) # Disable grid for correlation matrix

        self._set_labels(
            ax,
            xlabel=processed_params.get("xlabel"),
            ylabel=processed_params.get("ylabel"),
            subplot_title=processed_params.get("subplot_title", "Correlation Matrix"),
        )

        fig = self.apply_style_preset(fig)
        fig = self._apply_common_layout(fig, processed_params)
        return fig

    def plot_eigenvalues(self, criteria: Optional[List[str]] = None, **kwargs):
        if self.library != "matplotlib":
            print(
                f"plot_eigenvalues is currently implemented for Matplotlib. Skipping for {self.library}."
            )
            return None

        processed_params = self._process_common_params(**kwargs)
        fig, ax = self._create_figure(figsize=processed_params["figsize"])

        if not hasattr(self.dim_reduction_model, "V_ordered"):
            raise AttributeError("Model must have 'V_ordered' (ordered eigenvalues).")

        # --- Corrected section: Ensure numerical index for plotting ---
        original_model_index_exists = hasattr(self.dim_reduction_model, "index")
        original_model_index_value = getattr(self.dim_reduction_model, "index", None)

        # Always use a 1-based numerical index for plotting component numbers in this method
        numerical_plot_index = np.arange(1, len(self.dim_reduction_model.V_ordered) + 1)
        self.dim_reduction_model.index = numerical_plot_index
        # --- End of corrected section ---

        try:
            current_criteria = "all"
            if criteria:
                if "all" in [
                    c.lower() for c in criteria
                ]:  # Check for "all" keyword case-insensitively
                    current_criteria = "all"
                else:
                    current_criteria = [c.lower() for c in criteria]

            # Store requested criteria in the model temporarily for _plot_broken_stick_and_variance to access
            self.dim_reduction_model._plot_eigenvalues_criteria_requested = (
                current_criteria
            )

            main_plot_line = ax.plot(
                self.dim_reduction_model.index,  # Now guaranteed to be numerical_plot_index
                self.dim_reduction_model.V_ordered,
                marker="o",
                linestyle="-",
                color=self.colors["theme_color"],
                linewidth=2,
                label="Eigenvalues",
            )
            self._set_labels(
                ax,
                xlabel=processed_params.get("xlabel", "Component Number"),
                ylabel=processed_params.get("ylabel", "Eigenvalue"),
                subplot_title=processed_params.get("subplot_title", "Scree Plot"),
            )

            legend_handles = [main_plot_line[0]]

            if current_criteria:  # Check if there are any criteria to plot
                if "greater_than_one" in current_criteria or "all" in current_criteria:
                    h = self._plot_eigenvalues_greater_than_one(ax)
                    if h:
                        legend_handles.append(h)
                if (
                    "average_eigenvalue" in current_criteria
                    or "aec" in current_criteria
                ) or "all" in current_criteria:
                    h = self._plot_average_eigenvalue_criterion(ax)
                    if h:
                        legend_handles.append(h)
                if "kp" in current_criteria or "all" in current_criteria:
                    h = self._plot_KP_criterion(ax)
                    if h:
                        legend_handles.append(h)
                if "kl" in current_criteria or "all" in current_criteria:
                    h = self._plot_KL_criterion(ax)
                    if h:
                        legend_handles.append(h)
                if "caec" in current_criteria or "all" in current_criteria:
                    h = self._plot_CAEC_criterion(ax)
                    if h:
                        legend_handles.append(h)

                ax2_used = False
                # Check for broken_stick or variance_explained to set up secondary axis
                if (
                    "broken_stick" in current_criteria
                    or "variance_explained" in current_criteria
                ) or "all" in current_criteria:
                    ax2 = ax.twinx()
                    ax2_used = True
                    h_bs, h_var = self._plot_broken_stick_and_variance(
                        ax2
                    )  # Will use numerical index
                    if h_bs:
                        legend_handles.append(h_bs)
                    if h_var:
                        legend_handles.append(h_var)

                    ax2.set_ylabel("Percentage (%)", color=self.colors["text_color"])
                    ax2.tick_params(axis="y", labelcolor=self.colors["text_color"])
                    ax2.spines["right"].set_color(self.colors["text_color"])
                    ax2.grid(False)

                if (
                    "cumulative_variance" in current_criteria
                    or "all" in current_criteria
                ):
                    target_ax_for_cum_var = ax2 if ax2_used else ax.twinx()
                    if not ax2_used:
                        target_ax_for_cum_var.set_ylabel(
                            "Cumulative Variance (%)", color=self.colors["accent_color"]
                        )
                        target_ax_for_cum_var.tick_params(
                            axis="y", labelcolor=self.colors["accent_color"]
                        )
                        target_ax_for_cum_var.spines["right"].set_color(
                            self.colors["accent_color"]
                        )
                        target_ax_for_cum_var.grid(False)

                    h_cum_var = self._plot_cumulative_variance(
                        target_ax_for_cum_var
                    )  # Will use numerical index
                    if h_cum_var:
                        legend_handles.append(h_cum_var)
                    target_ax_for_cum_var.set_ylim(0, 105)

                if len(legend_handles) > 0:
                    legend_labels = [h.get_label() for h in legend_handles]
                    valid_handles_labels = [
                        (h, l)
                        for h, l in zip(legend_handles, legend_labels)
                        if l and not l.startswith("_")
                    ]
                    if valid_handles_labels:
                        valid_handles, valid_labels = zip(*valid_handles_labels)
                        ax.legend(
                            handles=list(valid_handles),
                            labels=list(valid_labels),
                            loc="best",
                        )
        finally:
            # Restore the original index attribute state
            if original_model_index_exists:
                self.dim_reduction_model.index = original_model_index_value
            else:
                # If 'index' did not exist before we set it, remove it.
                if hasattr(self.dim_reduction_model, "index"):  # Defensive check
                    del self.dim_reduction_model.index

            # Clean up temporary attribute for criteria (existing logic)
            if hasattr(
                self.dim_reduction_model, "_plot_eigenvalues_criteria_requested"
            ):
                del self.dim_reduction_model._plot_eigenvalues_criteria_requested

        fig = self.apply_style_preset(fig)
        fig = self._apply_common_layout(fig, processed_params)
        return fig

    def _plot_eigenvalues_greater_than_one(self, ax):
        if not hasattr(self.dim_reduction_model, "V_ordered"):
            return None
        # self.dim_reduction_model.index is now guaranteed to be numerical here
        num_gt_one = np.sum(self.dim_reduction_model.V_ordered >= 1)
        if num_gt_one > 0:
            # x-value calculation is now safe: number + float
            line = ax.axvline(
                x=(
                    self.dim_reduction_model.index[num_gt_one - 1] + 0.5
                    if num_gt_one > 0
                    else 0.5
                ),
                color=self.colors["highlight_color"],
                linestyle="--",
                label=r"Eigenvalues $\geq$ 1",
            )
            return line
        return None

    def _plot_average_eigenvalue_criterion(self, ax):
        if not hasattr(self.dim_reduction_model, "V_ordered"):
            return None
        # self.dim_reduction_model.index is now guaranteed to be numerical here
        mean_eig = self.dim_reduction_model.V_ordered.mean()
        num_aec = np.sum(self.dim_reduction_model.V_ordered >= mean_eig)
        if num_aec > 0:
            line = ax.axvline(
                x=(
                    self.dim_reduction_model.index[num_aec - 1] + 0.5
                    if num_aec > 0
                    else 0.5
                ),
                color=self.colors.get("detail_light_color", "grey"),
                linestyle=":",
                label="AEC (Avg Eigenvalue)",
            )
            return line
        return None

    def _plot_KP_criterion(self, ax):
        if not all(
            hasattr(self.dim_reduction_model, attr)
            for attr in ["V", "correlation_matrix"]
        ):
            return None
        # self.dim_reduction_model.index is now guaranteed to be numerical here
        V_orig = self.dim_reduction_model.V
        p = V_orig.size
        if p == 0:
            return None
        rank = np.linalg.matrix_rank(self.dim_reduction_model.correlation_matrix)
        if rank == 0:
            return None

        v_sorted_desc = np.sort(V_orig)[::-1]
        sum_term = sum(
            (v_sorted_desc[m] / v_sorted_desc.sum()) - (1 / p) for m in range(rank)
        )

        denominator = 2 * (p - 1) / p
        if denominator == 0:
            return None

        term_p_rank = (p - rank) * (1 / p) if p > 0 else 0

        val_inside_round = 1 + (p - 1) * (1 - (sum_term + term_p_rank) / denominator)
        x_kp = round(val_inside_round)
        if x_kp > 0 and x_kp <= p:
            line = ax.axvline(
                x=self.dim_reduction_model.index[int(x_kp) - 1] + 0.5,
                color=self.colors.get("prediction_band", "cyan"),
                linestyle="-.",
                label="KP Criterion",
            )
            return line
        return None

    def _plot_KL_criterion(self, ax):
        if not all(
            hasattr(self.dim_reduction_model, attr)
            for attr in ["V", "correlation_matrix"]
        ):
            return None
        # self.dim_reduction_model.index is now guaranteed to be numerical here
        V_orig = self.dim_reduction_model.V
        p = V_orig.size
        if p == 0:
            return None
        rank = np.linalg.matrix_rank(self.dim_reduction_model.correlation_matrix)
        if rank == 0:
            return None

        v_sorted_desc = np.sort(V_orig)[::-1]
        sum_term = sum(
            (v_sorted_desc[m] / v_sorted_desc.sum()) - (1 / p) for m in range(rank)
        )

        denominator = 2 * (p - 1) / p
        if denominator == 0:
            return None
        term_p_rank = (p - rank) * (1 / p) if p > 0 else 0

        exponent_val = 1 - (sum_term + term_p_rank) / denominator
        try:
            x_kl = round(p**exponent_val)
        except OverflowError:
            return None

        if x_kl > 0 and x_kl <= p:
            line = ax.axvline(
                x=self.dim_reduction_model.index[int(x_kl) - 1] + 0.5,
                color=self.colors.get("confidence_band", "orange"),
                linestyle=(0, (3, 1, 1, 1)),
                label="KL Criterion",
            )
            return line
        return None

    def _plot_CAEC_criterion(self, ax):
        if not hasattr(self.dim_reduction_model, "V_ordered"):
            return None
        # self.dim_reduction_model.index is now guaranteed to be numerical here
        threshold_val = 0.7 * self.dim_reduction_model.V_ordered.mean()
        num_caec = np.sum(self.dim_reduction_model.V_ordered >= threshold_val)
        if num_caec > 0:
            line = ax.axvline(
                x=self.dim_reduction_model.index[num_caec - 1] + 0.5,
                color=self.colors.get("detail_medium_color", "darkgrey"),
                linestyle=(0, (5, 5)),
                label="CAEC (70% Avg)",
            )
            return line
        return None

    def _plot_broken_stick_and_variance(self, ax_secondary):
        # self.dim_reduction_model.index is now guaranteed to be numerical here
        if not hasattr(self.dim_reduction_model, "V_ordered") or not hasattr(
            self.dim_reduction_model, "index"
        ):  # Should always have index due to main func
            return None, None

        p = len(self.dim_reduction_model.V_ordered)
        if p == 0:
            return None, None
        V_sum = np.sum(self.dim_reduction_model.V_ordered)
        if V_sum == 0:
            return None, None

        variance_explained_pct = (self.dim_reduction_model.V_ordered / V_sum) * 100

        bs_model_pct = np.zeros(p)
        for j_idx in range(p):
            bs_model_pct[j_idx] = (100.0 / p) * sum(
                1.0 / k_val for k_val in range(j_idx + 1, p + 1)
            )

        line_var, line_bs = None, None
        criteria_requested = getattr(
            self.dim_reduction_model, "_plot_eigenvalues_criteria_requested", []
        )

        if "variance_explained" in criteria_requested:
            line_var = ax_secondary.plot(
                self.dim_reduction_model.index,
                variance_explained_pct,
                color=self.colors["accent_color"],
                linestyle=":",
                marker="s",
                label="Variance Explained (%)",
            )[0]
        if "broken_stick" in criteria_requested:
            line_bs = ax_secondary.plot(
                self.dim_reduction_model.index,
                bs_model_pct,
                color=self.colors["theme_color"],
                linestyle="--",
                marker="^",
                label="Broken Stick Model (%)",
            )[0]

        max_y_val = 0
        if line_var is not None:
            max_y_val = max(
                max_y_val,
                (
                    np.max(variance_explained_pct)
                    if variance_explained_pct.size > 0
                    else 0
                ),
            )
        if line_bs is not None:
            max_y_val = max(
                max_y_val, np.max(bs_model_pct) if bs_model_pct.size > 0 else 0
            )
        ax_secondary.set_ylim(0, max(10, max_y_val * 1.1))

        return line_bs, line_var

    def _plot_cumulative_variance(self, ax_secondary):
        # self.dim_reduction_model.index is now guaranteed to be numerical here
        if not hasattr(self.dim_reduction_model, "V_ordered") or not hasattr(
            self.dim_reduction_model, "index"
        ):  # Should always have index
            return None
        V_sum = np.sum(self.dim_reduction_model.V_ordered)
        if V_sum == 0:
            return None

        cumulative_variance_pct = np.cumsum(
            (self.dim_reduction_model.V_ordered / V_sum) * 100
        )
        line = ax_secondary.plot(
            self.dim_reduction_model.index,
            cumulative_variance_pct,
            color=self.colors["accent_color"],
            linestyle="-.",
            marker="x",
            label="Cumulative Variance (%)",
        )[0]
        return line

    # ... (rest of the class methods remain unchanged) ...

    def plot_hotteling_t2_vs_q(self, show_legend=True, **kwargs):
        if self.library != "matplotlib":
            print(
                f"plot_hotteling_t2_vs_q is currently implemented for Matplotlib. Skipping for {self.library}."
            )
            return None

        processed_params = self._process_common_params(**kwargs)
        fig, ax = self._create_figure(figsize=processed_params["figsize"])

        if not all(hasattr(self.dim_reduction_model, attr) for attr in ["Q", "T2"]):
            raise AttributeError("Model missing Q or T2 attribute.")

        objects_labels_attr = getattr(self.dim_reduction_model, "objects", None)

        ax.scatter(
            self.dim_reduction_model.Q,
            self.dim_reduction_model.T2,
            color=self.colors["accent_color"],
            label="Observations",
        )

        if (
            objects_labels_attr is not None
            and hasattr(self.dim_reduction_model, "T2")
            and len(self.dim_reduction_model.T2) > 3
        ):
            t2_array = np.asarray(self.dim_reduction_model.T2)
            if t2_array.ndim == 1 and t2_array.size > 0:
                outlier_indices = np.argsort(t2_array)[-3:]
                q_array = np.asarray(self.dim_reduction_model.Q)
                for idx in outlier_indices:
                    if idx < len(objects_labels_attr) and idx < len(q_array):
                        ax.annotate(
                            objects_labels_attr[idx],
                            (q_array[idx], t2_array[idx]),
                            textcoords="offset points",
                            xytext=(0, 5),
                            ha="center",
                            fontsize=self.colors["annotation_fontsize_small"],
                        )

        self._set_labels(
            ax,
            xlabel=processed_params.get("xlabel", r"$Q$ (Squared Prediction Error)"),
            ylabel=processed_params.get("ylabel", r"Hotelling's $T^2$"),
            subplot_title=processed_params.get(
                "subplot_title", "Hotelling's T2 vs. Q residuals"
            ),
        )

        if show_legend and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")

        fig = self.apply_style_preset(fig)
        fig = self._apply_common_layout(fig, processed_params)
        return fig

    def plot_pci_contribution(self, **kwargs):
        if self.library != "matplotlib":
            print(
                f"plot_pci_contribution is currently implemented for Matplotlib. Skipping for {self.library}."
            )
            return None

        processed_params = self._process_common_params(**kwargs)
        fig, ax = self._create_figure(figsize=processed_params["figsize"])

        if not all(
            hasattr(self.dim_reduction_model, attr) for attr in ["W", "variables"]
        ):
            raise AttributeError("Model missing W (loadings) or variables attribute.")
        if not hasattr(self.dim_reduction_model, "n_variables"):
            self.dim_reduction_model.n_variables = self.dim_reduction_model.W.shape[0]

        num_components = self.dim_reduction_model.W.shape[1]
        cat_colors = self.colors["category_color_scale"]

        for i in range(num_components):
            ax.plot(
                np.arange(self.dim_reduction_model.n_variables),
                self.dim_reduction_model.W[:, i],
                marker="o",
                markerfacecolor="none",
                label=f"PC$_{i+1}$",
                color=cat_colors[i % len(cat_colors)],
            )

        self._set_labels(
            ax,
            xlabel=processed_params.get("xlabel", "Variable Index"),
            ylabel=processed_params.get("ylabel", "Loading Value"),
            subplot_title=processed_params.get(
                "subplot_title", "Contribution of Variables to PCs"
            ),
        )

        ax.set_xticks(np.arange(self.dim_reduction_model.n_variables))
        ax.set_xticklabels(self.dim_reduction_model.variables, rotation=45, ha="right")

        if processed_params.get("showlegend", True) and num_components > 0:
            ax.legend(loc="best")

        fig = self.apply_style_preset(fig)
        fig = self._apply_common_layout(fig, processed_params)
        return fig

    def _get_plot_components_indices(self, components_arg):
        if not hasattr(self.dim_reduction_model, "n_component"):
            if (
                hasattr(self.dim_reduction_model, "W")
                and self.dim_reduction_model.W is not None
            ):
                self.dim_reduction_model.n_component = self.dim_reduction_model.W.shape[
                    1
                ]
            elif (
                hasattr(self.dim_reduction_model, "T")
                and self.dim_reduction_model.T is not None
            ):
                self.dim_reduction_model.n_component = self.dim_reduction_model.T.shape[
                    1
                ]
            else:
                raise AttributeError(
                    "Model must have 'n_component' or 'W'/'T' attributes to determine it."
                )

        n_comp = self.dim_reduction_model.n_component

        if components_arg is None:
            return [(i, j) for i in range(n_comp) for j in range(n_comp)]
        elif isinstance(components_arg, tuple) and len(components_arg) == 2:
            i, j = components_arg
            if not (0 <= i < n_comp and 0 <= j < n_comp):
                raise ValueError(
                    f"Component indices ({i},{j}) out of bounds for {n_comp} components (0 to {n_comp-1})."
                )
            if i == j:
                raise ValueError(
                    f"Cannot plot single component against itself (PC{i+1} vs PC{j+1}). For matrix, diagonal is handled."
                )
            return [(i, j)]
        else:
            raise ValueError(
                "components argument must be a tuple (i,j) of 0-indexed components or None for matrix."
            )

    def plot_loadings(
        self, components: Optional[Tuple[int, int]] = None, show_arrows=True, **kwargs
    ):
        if self.library != "matplotlib":
            print(
                f"plot_loadings is currently implemented for Matplotlib. Skipping for {self.library}."
            )
            return None

        if not hasattr(self.dim_reduction_model, "W"):
            raise AttributeError("Model needs 'W' (loadings).")
        if not hasattr(self.dim_reduction_model, "n_component"):
            self.dim_reduction_model.n_component = self.dim_reduction_model.W.shape[1]
        if not hasattr(self.dim_reduction_model, "n_variables"):
            self.dim_reduction_model.n_variables = self.dim_reduction_model.W.shape[0]
        if not hasattr(self.dim_reduction_model, "variables"):
            self.dim_reduction_model.variables = [
                f"Var{k+1}" for k in range(self.dim_reduction_model.n_variables)
            ]

        is_matrix_plot = components is None
        plot_pairs_indices = self._get_plot_components_indices(components)

        processed_params = self._process_common_params(**kwargs)

        if not is_matrix_plot and len(plot_pairs_indices) == 1:
            idx_i, idx_j = plot_pairs_indices[0]
            fig, ax = self._create_figure(figsize=processed_params["figsize"])
            self._plot_loadings_on_axis(
                ax, idx_i, idx_j, show_arrows, is_matrix_plot=False
            )
            subplot_title = processed_params.get(
                "subplot_title", f"Loadings: PC$_{idx_i+1}$ vs PC$_{idx_j+1}$"
            )
            self._set_labels(ax, subplot_title=subplot_title)
            fig = self.apply_style_preset(fig)
            fig = self._apply_common_layout(fig, processed_params)
            return fig
        else:
            n_comp = self.dim_reduction_model.n_component
            if n_comp == 0:
                return None
            self._init_matplotlib_style()

            base_fig_w, base_fig_h = processed_params["figsize"]
            matrix_figsize = (
                min(base_fig_w * n_comp / 2.5, 20),
                min(base_fig_h * n_comp / 2.5, 20),
            )

            fig, axs = plt.subplots(
                n_comp, n_comp, figsize=matrix_figsize, squeeze=False
            )
            fig.set_facecolor(self.colors["bg_color"])

            matrix_title = processed_params.get("title", "Loadings Matrix")
            if matrix_title:
                fig.suptitle(
                    matrix_title,
                    fontsize=plt.rcParams["figure.titlesize"],
                    color=self.colors["text_color"],
                    weight=plt.rcParams["figure.titleweight"],
                )

            for r_idx in range(n_comp):
                for c_idx in range(n_comp):
                    ax = axs[r_idx, c_idx]
                    ax.set_facecolor(self.colors["bg_color"])
                    if r_idx == c_idx:
                        self._plot_empty_on_axis(ax, r_idx, "PC")
                    else:
                        self._plot_loadings_on_axis(
                            ax, c_idx, r_idx, show_arrows, is_matrix_plot=True
                        )

            fig = self._apply_common_layout(fig, processed_params)
            return fig

    def _plot_loadings_on_axis(
        self, ax, idx_i, idx_j, show_arrows=True, is_matrix_plot=False
    ):
        x_data = self.dim_reduction_model.W[:, idx_i]
        y_data = self.dim_reduction_model.W[:, idx_j]

        point_colors = getattr(
            self.dim_reduction_model, "variables_colors", self.colors["theme_color"]
        )
        variables = self.dim_reduction_model.variables

        ax.scatter(
            x_data, y_data, c=point_colors, s=30 if is_matrix_plot else 50, alpha=0.7
        )

        if show_arrows:
            for d_idx in range(self.dim_reduction_model.n_variables):
                ax.arrow(
                    0,
                    0,
                    x_data[d_idx],
                    y_data[d_idx],
                    length_includes_head=True,
                    width=0.01 if not is_matrix_plot else 0.005,
                    head_width=(
                        (0.03 if not is_matrix_plot else 0.015)
                        * (np.abs(x_data[d_idx]) + np.abs(y_data[d_idx]))
                        / 2
                        if (x_data[d_idx] != 0 or y_data[d_idx] != 0)
                        else (0.03 if not is_matrix_plot else 0.015)
                    ),  # Dynamic headwidth based on vector length
                    color=self.colors["arrow_color"],
                    alpha=0.6,
                )

        annot_fontsize = self.colors["annotation_fontsize_small"] - (
            2 if is_matrix_plot else 0
        )
        for d_idx in range(self.dim_reduction_model.n_variables):
            offset_scale = 1.1 if not is_matrix_plot else 1.15
            ha = "left" if x_data[d_idx] * offset_scale >= 0 else "right"
            va = "bottom" if y_data[d_idx] * offset_scale >= 0 else "top"

            ax.annotate(
                variables[d_idx],
                (x_data[d_idx] * offset_scale, y_data[d_idx] * offset_scale),
                ha=ha,
                va=va,
                color=self.colors["text_color"],
                fontsize=annot_fontsize,
            )

        self._set_labels(ax, xlabel=rf"PC$_{idx_i+1}$", ylabel=rf"PC$_{idx_j+1}$")
        self._apply_matplotlib_ax_style(
            ax, for_empty=False, is_matrix_plot=is_matrix_plot
        )

    def plot_scores(
        self, components: Optional[Tuple[int, int]] = None, label_points=False, **kwargs
    ):
        if self.library != "matplotlib":
            print(
                f"plot_scores is currently implemented for Matplotlib. Skipping for {self.library}."
            )
            return None

        if not hasattr(self.dim_reduction_model, "T"):
            raise AttributeError("Model needs 'T' (scores).")
        if not hasattr(self.dim_reduction_model, "n_component"):
            self.dim_reduction_model.n_component = self.dim_reduction_model.T.shape[1]
        if not hasattr(self.dim_reduction_model, "n_objects"):
            self.dim_reduction_model.n_objects = self.dim_reduction_model.T.shape[0]
        if not hasattr(self.dim_reduction_model, "objects"):
            self.dim_reduction_model.objects = [
                f"Obj{k+1}" for k in range(self.dim_reduction_model.n_objects)
            ]

        is_matrix_plot = components is None
        plot_pairs_indices = self._get_plot_components_indices(components)
        processed_params = self._process_common_params(**kwargs)

        if not is_matrix_plot and len(plot_pairs_indices) == 1:
            idx_i, idx_j = plot_pairs_indices[0]
            fig, ax = self._create_figure(figsize=processed_params["figsize"])
            self._plot_scores_on_axis(
                ax, idx_i, idx_j, label_points, is_matrix_plot=False
            )
            subplot_title = processed_params.get(
                "subplot_title", f"Scores: PC$_{idx_i+1}$ vs PC$_{idx_j+1}$"
            )
            self._set_labels(ax, subplot_title=subplot_title)
            if (
                processed_params.get("showlegend", True)
                and ax.get_legend_handles_labels()[0]
            ):
                objects_colors_attr = getattr(
                    self.dim_reduction_model, "objects_colors", None
                )
                if (
                    isinstance(objects_colors_attr, (list, np.ndarray))
                    and len(np.unique(objects_colors_attr)) > 1
                ):
                    unique_colors = []
                    unique_labels_for_legend = []
                    if self.dim_reduction_model.objects and len(
                        self.dim_reduction_model.objects
                    ) == len(objects_colors_attr):
                        temp_map = {}
                        for obj_idx, obj_col in enumerate(objects_colors_attr):
                            if obj_col not in temp_map:
                                temp_map[obj_col] = self.dim_reduction_model.objects[
                                    obj_idx
                                ]

                        for color_val, label_val in temp_map.items():
                            unique_colors.append(
                                patches.Patch(color=color_val, label=label_val)
                            )
                        if unique_colors:
                            ax.legend(handles=unique_colors, loc="best")
                    else:
                        ax.legend(loc="best")
                else:
                    ax.legend(loc="best")

            fig = self.apply_style_preset(fig)
            fig = self._apply_common_layout(fig, processed_params)
            return fig
        else:
            n_comp = self.dim_reduction_model.n_component
            if n_comp == 0:
                return None
            self._init_matplotlib_style()
            base_fig_w, base_fig_h = processed_params["figsize"]
            matrix_figsize = (
                min(base_fig_w * n_comp / 2.5, 20),
                min(base_fig_h * n_comp / 2.5, 20),
            )

            fig, axs = plt.subplots(
                n_comp, n_comp, figsize=matrix_figsize, squeeze=False
            )
            fig.set_facecolor(self.colors["bg_color"])
            matrix_title = processed_params.get("title", "Scores Matrix")
            if matrix_title:
                fig.suptitle(
                    matrix_title,
                    fontsize=plt.rcParams["figure.titlesize"],
                    color=self.colors["text_color"],
                    weight=plt.rcParams["figure.titleweight"],
                )

            for r_idx in range(n_comp):
                for c_idx in range(n_comp):
                    ax = axs[r_idx, c_idx]
                    ax.set_facecolor(self.colors["bg_color"])
                    if r_idx == c_idx:
                        self._plot_empty_on_axis(ax, r_idx, "PC")
                    else:
                        self._plot_scores_on_axis(
                            ax, c_idx, r_idx, label_points, is_matrix_plot=True
                        )

            objects_colors = getattr(self.dim_reduction_model, "objects_colors", None)
            if (
                processed_params.get("showlegend", True)
                and isinstance(objects_colors, (list, np.ndarray))
                and len(np.unique(objects_colors)) > 1
                and len(np.unique(objects_colors)) < 10
            ):
                handles = []
                if hasattr(self.dim_reduction_model, "object_categories_map"):
                    for (
                        color_val,
                        label_val,
                    ) in self.dim_reduction_model.object_categories_map.items():
                        if color_val not in [
                            h.get_facecolor()[0]
                            for h in handles
                            if hasattr(h, "get_facecolor")
                            and h.get_facecolor() is not None
                            and len(h.get_facecolor()) > 0
                        ]:
                            handles.append(
                                patches.Patch(color=color_val, label=label_val)
                            )
                elif (
                    isinstance(objects_colors, (list, np.ndarray))
                    and self.dim_reduction_model.objects
                ):
                    temp_map = {}
                    for obj_idx, obj_col in enumerate(objects_colors):
                        if isinstance(obj_col, np.ndarray):
                            obj_col_hashable = tuple(obj_col)
                        else:
                            obj_col_hashable = obj_col

                        if obj_col_hashable not in temp_map:
                            temp_map[obj_col_hashable] = (
                                self.dim_reduction_model.objects[obj_idx]
                            )
                    for color_val_hashable, label_val in temp_map.items():
                        handles.append(
                            patches.Patch(color=color_val_hashable, label=label_val)
                        )
                if handles:
                    fig.legend(
                        handles=handles,
                        loc="center left",
                        bbox_to_anchor=(1.0, 0.5),
                        bbox_transform=fig.transFigure,
                        ncol=1,
                    )
                    fig.subplots_adjust(right=0.85 if n_comp > 1 else 0.9)

            fig = self._apply_common_layout(fig, processed_params)
            return fig

    def _plot_scores_on_axis(
        self, ax, idx_i, idx_j, label_points=False, is_matrix_plot=False
    ):
        x_data = self.dim_reduction_model.T[:, idx_i]
        y_data = self.dim_reduction_model.T[:, idx_j]

        point_colors = getattr(
            self.dim_reduction_model, "objects_colors", self.colors["accent_color"]
        )
        objects = self.dim_reduction_model.objects

        ax.scatter(
            x_data,
            y_data,
            c=point_colors,
            s=20 if is_matrix_plot else 40,
            alpha=0.8,
            label=(
                "Observations"
                if not isinstance(point_colors, (list, np.ndarray))
                or len(np.unique(point_colors)) == 1
                else None
            ),
        )

        if label_points:
            annot_fontsize = self.colors["annotation_fontsize_small"] - (
                2 if is_matrix_plot else 0
            )
            for d_idx in range(self.dim_reduction_model.n_objects):
                obj_label = (
                    objects[d_idx][0]
                    if isinstance(objects[d_idx], (list, tuple))
                    else objects[d_idx]
                )
                ax.annotate(
                    obj_label,
                    (x_data[d_idx], y_data[d_idx]),
                    textcoords="offset points",
                    xytext=(5, 0),
                    ha="left",
                    color=self.colors["text_color"],
                    fontsize=annot_fontsize,
                )

        self._set_labels(ax, xlabel=rf"PC$_{idx_i+1}$", ylabel=rf"PC$_{idx_j+1}$")
        self._apply_matplotlib_ax_style(
            ax, for_empty=False, is_matrix_plot=is_matrix_plot
        )

    def plot_biplot(
        self, components: Optional[Tuple[int, int]] = None, label_points=False, **kwargs
    ):
        if self.library != "matplotlib":
            print(
                f"plot_biplot is currently implemented for Matplotlib. Skipping for {self.library}."
            )
            return None

        if not all(hasattr(self.dim_reduction_model, attr) for attr in ["T", "W"]):
            raise AttributeError("Model needs 'T' (scores) and 'W' (loadings).")
        if not hasattr(self.dim_reduction_model, "n_component"):
            self.dim_reduction_model.n_component = self.dim_reduction_model.T.shape[1]
        if not hasattr(self.dim_reduction_model, "n_variables"):
            self.dim_reduction_model.n_variables = self.dim_reduction_model.W.shape[0]
        if not hasattr(self.dim_reduction_model, "variables"):
            self.dim_reduction_model.variables = [
                f"Var{k+1}" for k in range(self.dim_reduction_model.n_variables)
            ]
        if not hasattr(self.dim_reduction_model, "n_objects"):
            self.dim_reduction_model.n_objects = self.dim_reduction_model.T.shape[0]
        if not hasattr(self.dim_reduction_model, "objects"):
            self.dim_reduction_model.objects = [
                f"Obj{k+1}" for k in range(self.dim_reduction_model.n_objects)
            ]

        is_matrix_plot = components is None
        plot_pairs_indices = self._get_plot_components_indices(components)
        processed_params = self._process_common_params(**kwargs)

        if not is_matrix_plot and len(plot_pairs_indices) == 1:
            idx_i, idx_j = plot_pairs_indices[0]
            fig, ax = self._create_figure(figsize=processed_params["figsize"])
            self._plot_biplot_on_axis(
                ax, idx_i, idx_j, label_points, is_matrix_plot=False
            )
            subplot_title = processed_params.get(
                "subplot_title", f"Biplot: PC$_{idx_i+1}$ vs PC$_{idx_j+1}$"
            )
            self._set_labels(ax, subplot_title=subplot_title)
            if processed_params.get("showlegend", True):
                score_color = getattr(
                    self.dim_reduction_model,
                    "objects_colors",
                    self.colors["accent_color"],
                )
                score_patch_color = (
                    score_color[0]
                    if isinstance(score_color, (list, np.ndarray))
                    else score_color
                )
                handles = [
                    patches.Patch(color=score_patch_color, label="Scores/Objects"),
                    plt.Line2D(
                        [0],
                        [0],
                        marker="",
                        color=self.colors["arrow_color"],
                        label="Loadings/Variables",
                        linestyle="-",
                        linewidth=2,
                    ),
                ]
                ax.legend(handles=handles, loc="best")

            fig = self.apply_style_preset(fig)
            fig = self._apply_common_layout(fig, processed_params)
            return fig
        else:
            n_comp = self.dim_reduction_model.n_component
            if n_comp == 0:
                return None
            self._init_matplotlib_style()
            base_fig_w, base_fig_h = processed_params["figsize"]
            matrix_figsize = (
                min(base_fig_w * n_comp / 2.5, 20),
                min(base_fig_h * n_comp / 2.5, 20),
            )

            fig, axs = plt.subplots(
                n_comp, n_comp, figsize=matrix_figsize, squeeze=False
            )
            fig.set_facecolor(self.colors["bg_color"])
            matrix_title = processed_params.get("title", "Biplot Matrix")
            if matrix_title:
                fig.suptitle(
                    matrix_title,
                    fontsize=plt.rcParams["figure.titlesize"],
                    color=self.colors["text_color"],
                    weight=plt.rcParams["figure.titleweight"],
                )

            for r_idx in range(n_comp):
                for c_idx in range(n_comp):
                    ax = axs[r_idx, c_idx]
                    ax.set_facecolor(self.colors["bg_color"])
                    if r_idx == c_idx:
                        self._plot_empty_on_axis(ax, r_idx, "PC")
                    else:
                        self._plot_biplot_on_axis(
                            ax, c_idx, r_idx, label_points, is_matrix_plot=True
                        )

            if processed_params.get("showlegend", True) and n_comp > 1:
                score_color_example = getattr(
                    self.dim_reduction_model,
                    "objects_colors",
                    self.colors["accent_color"],
                )
                if isinstance(score_color_example, (list, np.ndarray)):
                    score_color_example = score_color_example[0]

                handles = [
                    patches.Patch(color=score_color_example, label="Scores/Objects"),
                    plt.Line2D(
                        [0],
                        [0],
                        marker="",
                        color=self.colors["arrow_color"],
                        label="Loadings/Variables",
                        linestyle="-",
                        linewidth=1,
                    ),
                ]
                fig.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1.0, 0.5),
                    bbox_transform=fig.transFigure,
                    ncol=1,
                )
                fig.subplots_adjust(right=0.85)

            fig = self._apply_common_layout(fig, processed_params)
            return fig

    def _plot_biplot_on_axis(
        self, ax, idx_i, idx_j, label_points=False, is_matrix_plot=False
    ):
        scores_x = self.dim_reduction_model.T[:, idx_i]
        scores_y = self.dim_reduction_model.T[:, idx_j]
        objects_colors = getattr(
            self.dim_reduction_model, "objects_colors", self.colors["accent_color"]
        )
        objects = self.dim_reduction_model.objects

        score_ptp_x = np.ptp(scores_x) if scores_x.size > 1 else 1.0
        score_ptp_y = np.ptp(scores_y) if scores_y.size > 1 else 1.0
        score_ptp_x = max(score_ptp_x, 1e-6)
        score_ptp_y = max(score_ptp_y, 1e-6)

        loading_max_abs_x = (
            np.max(np.abs(self.dim_reduction_model.W[:, idx_i]))
            if self.dim_reduction_model.W[:, idx_i].size > 0
            else 1.0
        )
        loading_max_abs_y = (
            np.max(np.abs(self.dim_reduction_model.W[:, idx_j]))
            if self.dim_reduction_model.W[:, idx_j].size > 0
            else 1.0
        )
        loading_max_abs_x = max(loading_max_abs_x, 1e-6)
        loading_max_abs_y = max(loading_max_abs_y, 1e-6)

        scale_x = (score_ptp_x / loading_max_abs_x) * 0.33
        scale_y = (score_ptp_y / loading_max_abs_y) * 0.33

        ax.scatter(
            scores_x,
            scores_y,
            c=objects_colors,
            s=15 if is_matrix_plot else 30,
            alpha=0.6,
        )

        if label_points:
            annot_fontsize = self.colors["annotation_fontsize_small"] - (
                3 if is_matrix_plot else 1
            )
            for d_idx in range(self.dim_reduction_model.n_objects):
                obj_label = (
                    objects[d_idx][0]
                    if isinstance(objects[d_idx], (list, tuple))
                    else objects[d_idx]
                )
                ax.annotate(
                    obj_label,
                    (scores_x[d_idx], scores_y[d_idx]),
                    textcoords="offset points",
                    xytext=(3, 0),
                    ha="left",
                    color=self.colors["text_color"],
                    fontsize=annot_fontsize,
                )

        loadings_x_s = self.dim_reduction_model.W[:, idx_i] * scale_x
        loadings_y_s = self.dim_reduction_model.W[:, idx_j] * scale_y
        variables = self.dim_reduction_model.variables

        arrow_color_biplot = self.colors.get(
            "detail_medium_color", self.colors["text_color"]
        )

        for d_idx in range(self.dim_reduction_model.n_variables):
            arr_x, arr_y = loadings_x_s[d_idx], loadings_y_s[d_idx]
            if abs(arr_x) < 1e-6 and abs(arr_y) < 1e-6:
                continue

            ax.arrow(
                0,
                0,
                arr_x,
                arr_y,
                head_width=(
                    0.05 * (abs(arr_x) + abs(arr_y)) / 2
                    if not is_matrix_plot
                    else 0.03 * (abs(arr_x) + abs(arr_y)) / 2
                ),
                fc=arrow_color_biplot,
                ec=arrow_color_biplot,
                alpha=0.75,
                length_includes_head=True,
            )

            ha = "left" if arr_x >= 0 else "right"
            va = "bottom" if arr_y >= 0 else "top"
            annot_fontsize_load = self.colors["annotation_fontsize_small"] - (
                2 if is_matrix_plot else 0
            )
            ax.annotate(
                variables[d_idx],
                (arr_x * 1.1, arr_y * 1.1),
                ha=ha,
                va=va,
                color=arrow_color_biplot,
                fontsize=annot_fontsize_load,
                fontweight="bold",
            )

        self._set_labels(ax, xlabel=rf"PC$_{idx_i+1}$", ylabel=rf"PC$_{idx_j+1}$")
        self._apply_matplotlib_ax_style(
            ax, for_empty=False, is_matrix_plot=is_matrix_plot
        )

    def plot_explained_variance_ellipse(
        self, components=(0, 1), confidence_level=0.95, **kwargs
    ):
        if self.library != "matplotlib":
            print(
                f"plot_explained_variance_ellipse is currently implemented for Matplotlib. Skipping for {self.library}."
            )
            return None

        processed_params = self._process_common_params(**kwargs)
        fig, ax = self._create_figure(figsize=processed_params["figsize"])
        idx_i, idx_j = components

        if not hasattr(self.dim_reduction_model, "T"):
            raise AttributeError("Model must have 'T' (scores).")
        if not (
            0 <= idx_i < self.dim_reduction_model.T.shape[1]
            and 0 <= idx_j < self.dim_reduction_model.T.shape[1]
        ):
            raise ValueError(
                f"Component indices ({idx_i},{idx_j}) invalid for scores matrix."
            )

        x_data = self.dim_reduction_model.T[:, idx_i]
        y_data = self.dim_reduction_model.T[:, idx_j]
        point_colors = getattr(
            self.dim_reduction_model, "objects_colors", self.colors["accent_color"]
        )

        ax.scatter(x_data, y_data, c=point_colors, label="Data Points")

        if x_data.size > 1 and y_data.size > 1:
            cov_matrix = np.cov(x_data, y_data)
            if np.all(np.linalg.eigvals(cov_matrix) > 1e-9):
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]

                angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                angle_deg = np.degrees(angle_rad)

                s = chi2.ppf(confidence_level, df=2)

                width = 2 * np.sqrt(s * eigenvalues[0])
                height = 2 * np.sqrt(s * eigenvalues[1])

                ellipse = Ellipse(
                    xy=(np.mean(x_data), np.mean(y_data)),
                    width=width,
                    height=height,
                    angle=angle_deg,
                    facecolor="none",
                    edgecolor=self.colors["ellipse_color"],
                    linestyle="--",
                    linewidth=2,
                    label=f"{confidence_level*100:.0f}% Conf. Ellipse",
                )
                ax.add_patch(ellipse)
            else:
                print(
                    f"Warning: Covariance matrix for components ({idx_i+1}, {idx_j+1}) is singular or not positive definite. Cannot draw ellipse."
                )
        else:
            print("Warning: Not enough data points to draw ellipse.")

        self._set_labels(
            ax,
            xlabel=processed_params.get("xlabel", f"PC$_{idx_i+1}$"),
            ylabel=processed_params.get("ylabel", f"PC$_{idx_j+1}$"),
            subplot_title=processed_params.get(
                "subplot_title", "Scores with Confidence Ellipse"
            ),
        )

        if (
            processed_params.get("showlegend", True)
            and ax.get_legend_handles_labels()[0]
        ):
            ax.legend(loc="best")

        fig = self.apply_style_preset(fig)
        fig = self._apply_common_layout(fig, processed_params)
        return fig

    def _plot_empty_on_axis(self, ax, index, prefix="PC"):
        ax.text(
            0.5,
            0.5,
            f"{prefix}$_{index+1}$",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=plt.rcParams["axes.labelsize"] + (2 if prefix == "PC" else 4),
            color=self.colors["text_color"],
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self._apply_matplotlib_ax_style(ax, for_empty=True, is_matrix_plot=True)

    def _apply_matplotlib_ax_style(self, ax, for_empty=False, is_matrix_plot=False):
        if self.library != "matplotlib":
            return

        preset_name = self.style_preset
        preset_settings = self.STYLE_PRESETS["matplotlib"].get(preset_name, {})

        grid_enabled = preset_settings.get("axes.grid", True) and not for_empty
        if grid_enabled:
            ax.grid(
                grid_enabled,
                alpha=preset_settings.get("grid.alpha", 0.5),
                color=self.colors["grid_color"],
            )
        else:
            ax.grid(grid_enabled)

        ax.spines["top"].set_visible(
            preset_settings.get("axes.spines.top", True) and not for_empty
        )
        ax.spines["right"].set_visible(
            preset_settings.get("axes.spines.right", True) and not for_empty
        )
        ax.spines["left"].set_visible(not for_empty)
        ax.spines["bottom"].set_visible(not for_empty)

        for spine_pos in ["top", "right", "left", "bottom"]:
            if ax.spines[spine_pos].get_visible():
                ax.spines[spine_pos].set_color(self.colors["text_color"])

        current_xtick_size_param = plt.rcParams["xtick.labelsize"]
        current_ytick_size_param = plt.rcParams["ytick.labelsize"]

        final_xtick_labelsize = current_xtick_size_param
        final_ytick_labelsize = current_ytick_size_param

        reduction_amount = 2

        if is_matrix_plot:
            if (
                isinstance(current_xtick_size_param, (int, float))
                and current_xtick_size_param > 10
            ):
                final_xtick_labelsize = current_xtick_size_param - reduction_amount

            if (
                isinstance(current_ytick_size_param, (int, float))
                and current_ytick_size_param > 10
            ):
                final_ytick_labelsize = current_ytick_size_param - reduction_amount

        ax.tick_params(
            axis="x",
            colors=self.colors["text_color"],
            labelsize=final_xtick_labelsize,
            pad=5,
        )
        ax.tick_params(
            axis="y",
            colors=self.colors["text_color"],
            labelsize=final_ytick_labelsize,
            pad=5,
        )
