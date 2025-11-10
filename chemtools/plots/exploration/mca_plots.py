import matplotlib.pyplot as plt
import numpy as np
from ..base import BasePlotter

class MCAPlots(BasePlotter):
    """Class to generate various plots related to Multiple Correspondence Analysis (MCA)."""

    def __init__(self, mca_object, **kwargs):
        kwargs.setdefault("theme", "classic_professional_light")
        super().__init__(**kwargs)
        self.mca_object = mca_object

    def plot_eigenvalues(self, **kwargs):
        """Plots eigenvalues of the MCA."""
        if "plotter_kwargs" in kwargs:
            plotter_specific_kwargs = kwargs.pop("plotter_kwargs")
            temp_plotter = MCAPlots(self.mca_object, **plotter_specific_kwargs)
            return temp_plotter.plot_eigenvalues(**kwargs)

        params = self._process_common_params(**kwargs)
        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            ax.plot(self.mca_object.V_ordered, marker="o", linestyle="-", color=self.colors["theme_color"])
            self._set_labels(ax, subplot_title=params.get("subplot_title", "Scree Plot - Eigenvalues"), xlabel="Principal Components", ylabel="Eigenvalue")
            self._apply_common_layout(fig, params)
            return fig
        elif self.library == "plotly":
            import plotly.express as px
            title = params.get('title', 'Scree Plot - Eigenvalues')
            fig = px.line(x=np.arange(len(self.mca_object.V_ordered)), y=self.mca_object.V_ordered, title=title, labels={'x':'Principal Components', 'y':'Eigenvalue'}, markers=True, color_discrete_sequence=[self.colors['theme_color']])
            self._apply_common_layout(fig, params)
            return fig

    def plot_objects(self, axes=[0, 1], **kwargs):
        """Plots objects on the first two principal components."""
        if "plotter_kwargs" in kwargs:
            plotter_specific_kwargs = kwargs.pop("plotter_kwargs")
            temp_plotter = MCAPlots(self.mca_object, **plotter_specific_kwargs)
            return temp_plotter.plot_objects(axes=axes, **kwargs)

        params = self._process_common_params(**kwargs)
        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            ax.scatter(
                self.mca_object.L_ordered[:, axes[0]],
                self.mca_object.L_ordered[:, axes[1]],
                c=self.mca_object.objects_colors,
            )
            for i, txt in enumerate(self.mca_object.objects):
                ax.annotate(
                    txt, (self.mca_object.L_ordered[i, axes[0]], self.mca_object.L_ordered[i, axes[1]])
                )
            self._set_labels(ax, subplot_title=params.get("subplot_title", "Objects Plot"), xlabel=f"PC{axes[0]+1}", ylabel=f"PC{axes[1]+1}")
            self._apply_common_layout(fig, params)
            return fig
        elif self.library == "plotly":
            import plotly.express as px
            title = params.get('title', 'Objects Plot')
            fig = px.scatter(x=self.mca_object.L_ordered[:, axes[0]], y=self.mca_object.L_ordered[:, axes[1]],
                             color=self.mca_object.objects_colors, text=self.mca_object.objects, title=title,
                             labels={'x':f'PC{axes[0]+1}', 'y':f'PC{axes[1]+1}'})
            fig.update_traces(textposition='top center')
            self._apply_common_layout(fig, params)
            return fig
