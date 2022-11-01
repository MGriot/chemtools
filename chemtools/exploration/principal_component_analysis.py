import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from chemtools.preprocessing import autoscaling
from chemtools.preprocessing import correlation_matrix
from chemtools.preprocessing import diagonalized_matrix
from chemtools.utility import reorder_array
from chemtools.utility import heatmap
from chemtools.utility import annotate_heatmap
from chemtools.utility import random_colorHEX
from chemtools.utility import directory_creator
from chemtools.utility import when_date
from chemtools.utility import matplotlib_savefig


# maca Hotelling's T-Squared e T2 vs Q plot e upper control limit (UCL) for T
class principal_component_analysis:
    def __init__(
        self,
        x,
        sample_name=None,
        sample_name_row=None,
        variable_name_column=[],
        variable_name=None,
        text_color="black",
    ):
        if isinstance(x, pd.DataFrame):
            df = x
            self.sample_name = df[sample_name_row].to_numpy()
            # riga che contiene i nomi delle variabili saltando la prima cella, contiene il valore "label" che è dei campioni
            self.variable = df.columns[
                variable_name_column[0] : variable_name_column[1]
            ].to_numpy()
            self.x = df[self.variable].to_numpy()
        else:
            self.x = x
            self.sample_name = sample_name
            self.variable = variable_name
        self.text_color = text_color
        self.variable_number = len(self.variable)
        self.sample_number = len(self.sample_name)
        self.variable_color = random_colorHEX(self.variable.shape[0])
        self.sample_color = random_colorHEX(self.sample_name.shape[0])
        self.autoscaled = autoscaling(self.x)
        self.correlation_matrix = correlation_matrix(self.autoscaled)
        # eighenvalue enighenvector
        self.V, self.L = diagonalized_matrix(self.correlation_matrix)
        V_ordered, order = reorder_array(self.V)
        # riorino le PCi
        PC_index = np.zeros((3, self.L.shape[1]), dtype=object)
        for i in range(self.V.shape[0]):
            PC = f"PC{i+1}"
            PC_index[0, i] = PC
            PC_index[1, i] = order[i]
            PC_index[2, i] = V_ordered[i]
        self.PC_index = PC_index

        self.number_of_PC = self.L.shape[1]

    # matrice score -> matrix reduction
    @property
    def L_r(self):
        L_r = np.zeros((self.number_of_PC, self.variable_number))
        for aa, i in enumerate(range(self.number_of_PC)):
            # L_r[:,aa]=L[i]
            L_r[aa, :] = self.L[:, i]
        return L_r.transpose()

    @property
    def T_r(self):
        L_r = np.zeros((self.number_of_PC, self.variable_number))
        for aa, i in enumerate(range(self.number_of_PC)):
            # L_r[:,aa]=L[i]
            L_r[aa, :] = self.L[:, i]
        return np.dot(self.autoscaled, L_r.transpose())

    @property
    def E(self):
        T_r = np.pad(
            self.T_r,
            [
                (0, self.x.shape[0] - self.T_r.shape[0]),
                (0, self.x.shape[1] - self.T_r.shape[1]),
            ],
            mode="constant",
        )
        return np.subtract(self.x, T_r)

    def correlation_plot(
        self,
        savefig=False,
        DPI=None,
        output="output",
        name="correlation_plot",
        fig_format="png",
        transparent_background=True,
    ):
        # plot correlazione variabili
        fig, axs = plt.subplots(
            self.autoscaled.shape[1],
            self.autoscaled.shape[1],
            figsize=(5 * self.autoscaled.shape[1], 5 * self.autoscaled.shape[1]),
        )
        a = 0
        b = 0
        for i in range(self.autoscaled.shape[1]):
            a += 1
            a = 1
            for j in range(self.autoscaled.shape[1]):
                plt.subplot(self.autoscaled.shape[1], self.autoscaled.shape[1], a + b)
                plt.scatter(self.autoscaled[i], self.autoscaled[j])
                plt.xlabel(self.variable[i])
                plt.ylabel(self.variable[j])
                b += 1
        plt.tight_layout()
        matplotlib_savefig(
            savefig=savefig,
            DPI=DPI,
            output=output,
            name=name,
            fig_format=fig_format,
            transparent_background=transparent_background,
        )

    def correlation_matrix_plot(
        self,
        cmap="coolwarm",
        threshold=None,
        savefig=False,
        DPI=None,
        output="output",
        name="correlation_matrix_plot",
        fig_format="png",
        transparent_background=True,
    ):
        """_summary_

        Args:
            cmap (str, optional): Colormap color. Defaults to "coolwarm".
                ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys',
                'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','binary',
                'gist_yarg', 'gist_gray', 'gray', 'bone',
                'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper','PiYG', 'PRGn',
                'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                'twilight', 'twilight_shifted', 'hsv','Pastel1', 'Pastel2',
                'Paired', 'Accent', 'Dark2',
                'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                'tab20c','flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                'turbo', 'nipy_spectral', 'gist_ncar']
            threshold (_type_, optional): Value in data units according to which the colors
                from textcolors are applied.  If None (the default) uses the middle of the
                colormap as separation.  Optional.
            pfig (bool, optional): _description_. Defaults to False.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        im, cbar = heatmap(
            self.correlation_matrix,
            self.variable,
            self.variable,
            ax=ax,
            cmap=cmap,
            cbarlabel="Correlation value",
        )
        texts = annotate_heatmap(im, threshold=threshold, valfmt="{x:.2f}")
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        matplotlib_savefig(
            savefig=savefig,
            DPI=DPI,
            output=output,
            name=name,
            fig_format=fig_format,
            transparent_background=transparent_background,
        )

    # da togliere tutti i riferimenti ai dataframe e convertire in matrice
    def eigenvalues_plot(
        self,
        savefig=False,
        DPI=None,
        output="output",
        name="eigenvalues_plot",
        fig_format="png",
        transparent_background=True,
    ):
        """mancano:
        * KP
        * KL

        da libreria matlab"""
        plt.figure(figsize=(self.PC_index.shape[1], 5))
        # grafico degli autovalori maggiori di 1
        # da verificare un attimo se non rimane uguale all'autovalore medio
        plt.axvline(
            x=np.argmax(self.PC_index[2, :] < 1),
            color="brown",
            linestyle="-",
            label="Autovalori maggiori di 1",
        )
        # grafico degli autovalori
        plt.bar(
            x=self.PC_index[0, :],
            height=(self.PC_index[2, :] / self.PC_index[2, :].sum()) * 100,
            fill=False,
            edgecolor="darkorange",
            label="Varianza %",
        )

        # varianza cumulata
        y = 0
        c = np.array([])
        for i in self.PC_index[1, :]:
            y += (self.PC_index[2, i] / self.PC_index[2, :].sum()) * 100
            c = np.append(c, y)
        plt.bar(
            x=self.PC_index[0, :],
            height=c,
            fill=False,
            edgecolor="black",
            linestyle="--",
            width=0.6,
            label="Varanza cumulata %",
        )

        # criterio dell'autovalore medio
        plt.axvline(
            x=np.argmax(self.PC_index[2, :] < self.PC_index[2, :].mean()) - 1,
            color="red",
            alpha=0.5,
            linestyle="-",
            label="AEC",
        )

        # criterio KP
        plt.axvline(
            x=round(
                1
                + (self.V.size - 1)
                * (
                    1
                    - (
                        (
                            (
                                sum(
                                    self.V[m] / self.V.sum() - 1 / self.V.size
                                    for m in range(
                                        np.linalg.matrix_rank(self.correlation_matrix)
                                    )
                                )
                            )
                            + (
                                self.V.size
                                - np.linalg.matrix_rank(self.correlation_matrix)
                            )
                            ** (1 / self.V.size)
                        )
                        / (2 * (self.V.size - 1) / self.V.size)
                    )
                )
            )
            - 1,
            color="purple",
            alpha=0.5,
            linestyle="--",
            label="KP",
        )

        # criterio KL
        plt.axvline(
            x=round(
                self.V.size
                ** (
                    1
                    - (
                        sum(
                            self.V[m] / self.V.sum() - 1 / self.V.size
                            for m in range(
                                np.linalg.matrix_rank(self.correlation_matrix)
                            )
                        )
                        + (self.V.size - np.linalg.matrix_rank(self.correlation_matrix))
                        ** (1 / self.V.size)
                    )
                    / (2 * (self.V.size - 1) / self.V.size)
                )
            )
            - 1,
            color="cyan",
            alpha=0.5,
            linestyle="-",
            label="KL",
        )

        # criterio dell'autovalore medio *0.7
        plt.axvline(
            x=np.argmax(self.PC_index[2, :] < 0.7 * self.PC_index[2, :].mean()),
            color="blue",
            alpha=0.5,
            linestyle="--",
            label="CAEC",
        )

        # broken stick
        a = 1
        dm = np.array([])
        for i in range(0, self.PC_index.shape[1]):
            m = (100 / self.PC_index.shape[1]) * np.sum(
                1 / self.PC_index[1, a : self.PC_index.shape[1]]
            )
            dm = np.append(dm, m)
            a += 1
        plt.plot(self.PC_index[0, :], dm, color="lightgreen", label="Broken stick")

        plt.title("Eigenvalue")
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend()
        # plt.legend(["autovalori maggiori di 1","varianza %","Varanza cumulata%","AEC","CAEC"])
        plt.tight_layout()
        matplotlib_savefig(
            savefig=savefig,
            DPI=DPI,
            output=output,
            name=name,
            fig_format=fig_format,
            transparent_background=transparent_background,
        )

    def pci_contribution_plot(
        self,
        text_color="black",
        savefig=False,
        DPI=None,
        output="output",
        name="pci_contribution_plot",
        fig_format="png",
        transparent_background=True,
    ):
        for i in range(self.L_r.shape[1]):
            plt.plot(
                np.arange(self.variable.shape[0]),
                self.L_r[:, i],
                marker="o",
                markerfacecolor="none",
                label=f"PC{i+1}",
            )
        plt.title("Contributions of the PCi")
        plt.xticks(np.arange(self.variable.shape[0]), self.variable)
        plt.legend(labelcolor=text_color)
        plt.xlabel("Variable")
        plt.ylabel("Value of loading")
        matplotlib_savefig(
            savefig=savefig,
            DPI=DPI,
            output=output,
            name=name,
            fig_format=fig_format,
            transparent_background=transparent_background,
        )

    def loadings_plot(
        self,
        arrows=True,
        savefig=False,
        DPI=None,
        output="output",
        name="loadings_plot",
        fig_format="png",
        transparent_background=True,
    ):
        # Loadings plot
        fig, axs = plt.subplots(
            self.number_of_PC,
            self.number_of_PC,
            figsize=(5 * self.number_of_PC, 5 * self.number_of_PC),
        )
        a = 1
        for i in range(self.number_of_PC):
            for j in range(self.number_of_PC):
                fig.suptitle("Loadings plot", fontsize=24, y=1, color=self.text_color)
                ax = plt.subplot(self.number_of_PC, self.number_of_PC, a)
                if i != j:
                    for d in range(self.x.shape[1]):
                        #######################
                        if self.L_r[d, i] > self.L_r[:, i].mean():
                            if self.L_r[d, j] > self.L_r[:, j].mean():
                                position = ["left", "bottom"]
                            elif self.L_r[d, j] < self.L_r[:, j].mean():
                                position = ["right", "top"]
                        elif self.L_r[d, i] < self.L_r[:, i].mean():
                            if self.L_r[d, j] > self.L_r[:, j].mean():
                                position = ["left", "bottom"]
                            elif self.L_r[d, j] < self.L_r[:, j].mean():
                                position = ["right", "top"]
                        ######################
                        ax.scatter(
                            x=self.L_r[d, i],
                            y=self.L_r[d, j],
                            label=self.variable[d],
                            color=self.variable_color[d],
                        )
                        if arrows == True:
                            ax.arrow(
                                0,
                                0,
                                self.L_r[d, i],
                                self.L_r[d, j],
                                length_includes_head=True,
                                head_width=0.015,
                                color=self.text_color,
                                alpha=0.3,
                            )
                        ax.annotate(
                            text=self.variable[d],
                            xy=(self.L_r[d, i], self.L_r[d, j]),
                            ha=position[0],
                            va=position[1],
                            color=self.text_color,
                        )
                        ax.set_xlabel(f"PC{i+1}")
                        ax.set_ylabel(f"PC{j+1}")
                        # ax.legend()
                    handles, labels = ax.get_legend_handles_labels()
                    # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5),labelcolor=text_color, framealpha=0.1)
                    # fig.legend(labels=variable)
                if i == j:
                    # build a square in axes coords
                    left, width = 0, 1
                    bottom, height = 0, 1
                    right = left + width
                    top = bottom + height
                    # axes coordinates: (0, 0) is bottom left and (1, 1) is upper right
                    p = patches.Rectangle(
                        (left, bottom),
                        width,
                        height,
                        fill=False,
                        transform=ax.transAxes,
                        clip_on=False,
                    )
                    ax.text(
                        0.5 * (left + right),
                        0.5 * (bottom + top),
                        f"PC{i+1}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=20,
                        color=self.text_color,
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

                a += 1
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=self.text_color,
        )
        plt.tight_layout()
        matplotlib_savefig(
            savefig=savefig,
            DPI=DPI,
            output=output,
            name=name,
            fig_format=fig_format,
            transparent_background=transparent_background,
        )

    def scores_plot(
        self,
        label_point=False,
        savefig=False,
        DPI=None,
        output="output",
        name="scores_plot",
        fig_format="png",
        transparent_background=True,
    ):
        # Scores plot
        fig, axs = plt.subplots(
            self.number_of_PC,
            self.number_of_PC,
            figsize=(5 * self.number_of_PC, 5 * self.number_of_PC),
        )
        a = 1
        for i in range(self.number_of_PC):
            for j in range(self.number_of_PC):
                fig.tight_layout()
                fig.suptitle("Scores plot", fontsize=24, y=1, color=self.text_color)
                ax = plt.subplot(self.number_of_PC, self.number_of_PC, a)
                if i != j:
                    for d in range(self.x.shape[0]):
                        # ax.scatter(x=T_r[d,i], y=T_r[d,j], c=sample_color[d], edgecolors='black', label=sample[d])
                        ax.scatter(
                            x=self.T_r[d, i],
                            y=self.T_r[d, j],
                            c=self.sample_color[d],
                            label=self.sample_name[d, 0],
                        )
                        if label_point == True:
                            try:
                                ax.annotate(
                                    text=self.sample_name[d, 0][0],
                                    xy=(self.T_r[d, i], self.T_r[d, j]),
                                    color=self.text_color,
                                )
                            except:
                                pass
                        ax.set_xlabel(f"PC{i+1}")
                        ax.set_ylabel(f"PC{j+1}")
                        # ax.legend()
                    handles, labels = ax.get_legend_handles_labels()
                    # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), labelcolor=text_color, framealpha=0.1)
                if i == j:
                    # build a square in axes coords
                    left, width = 0, 1
                    bottom, height = 0, 1
                    right = left + width
                    top = bottom + height
                    # axes coordinates: (0, 0) is bottom left and (1, 1) is upper right
                    p = patches.Rectangle(
                        (left, bottom),
                        width,
                        height,
                        fill=False,
                        transform=ax.transAxes,
                        clip_on=False,
                    )
                    ax.text(
                        0.5 * (left + right),
                        0.5 * (bottom + top),
                        f"PC{i+1}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=20,
                        color=self.text_color,
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                a += 1
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=self.text_color,
        )
        plt.tight_layout
        matplotlib_savefig(
            savefig=savefig,
            DPI=DPI,
            output=output,
            name=name,
            fig_format=fig_format,
            transparent_background=transparent_background,
        )

    def biplots(
        self,
        label_point=False,
        subplot_dimensions=[5, 5],
        savefig=False,
        DPI=None,
        output="output",
        name="biplots",
        fig_format="png",
        transparent_background=True,
    ):
        # biplots
        fig, axs = plt.subplots(
            self.number_of_PC,
            self.number_of_PC,
            figsize=(
                subplot_dimensions[0] * self.number_of_PC,
                subplot_dimensions[1] * self.number_of_PC,
            ),
        )
        fig.suptitle("Biplots plot", fontsize=24, y=1, color=self.text_color)
        a = 1
        for i in range(self.number_of_PC):
            for j in range(self.number_of_PC):
                ax = plt.subplot(self.number_of_PC, self.number_of_PC, a)
                if i != j:
                    for e in range(self.x.shape[0]):
                        # ax.scatter(x=T_r[e,i], y=T_r[e,j], c=sample_color[e], edgecolors='black', label=sample[e])
                        ax.scatter(
                            x=self.T_r[e, i],
                            y=self.T_r[e, j],
                            c=self.sample_color[e],
                            label=self.sample_name[e, 0],
                        )
                        if label_point == True:
                            try:
                                ax.annotate(
                                    text=self.sample_name[d, 0][0],
                                    xy=(self.T_r[d, i], self.T_r[d, j]),
                                    color=self.text_color,
                                )
                            except:
                                pass
                    for d in range(self.x.shape[1]):
                        #######################
                        if self.L_r[d, i] > self.L_r[:, i].mean():
                            if self.L_r[d, j] > self.L_r[:, j].mean():
                                position = ["left", "bottom"]
                            elif self.L_r[d, j] < self.L_r[:, j].mean():
                                position = ["right", "top"]
                        elif self.L_r[d, i] < self.L_r[:, i].mean():
                            if self.L_r[d, j] > self.L_r[:, j].mean():
                                position = ["left", "bottom"]
                            elif self.L_r[d, j] < self.L_r[:, j].mean():
                                position = ["right", "top"]
                        ######################
                        ax.arrow(
                            0,
                            0,
                            pow(self.T_r[:, i] ** 2, 0.5).max() * self.L_r[d, i],
                            pow(self.T_r[:, j] ** 2, 0.5).max() * self.L_r[d, j],
                            length_includes_head=True,
                            head_width=0.15,
                            color=self.text_color,
                            alpha=0.3,
                        )
                        ax.annotate(
                            text=self.variable[d],
                            xy=(
                                pow(self.T_r[:, i] ** 2, 0.5).max() * self.L_r[d, i],
                                pow(self.T_r[:, j] ** 2, 0.5).max() * self.L_r[d, j],
                            ),
                            ha=position[0],
                            va=position[1],
                            color=self.text_color,
                        )
                    ax.set_xlabel(f"PC{i+1}")
                    ax.set_ylabel(f"PC{j+1}")
                    handles, labels = ax.get_legend_handles_labels()
                    # ax.legend()
                if i == j:
                    # build a square in axes coords
                    left, width = 0, 1
                    bottom, height = 0, 1
                    right = left + width
                    top = bottom + height
                    # axes coordinates: (0, 0) is bottom left and (1, 1) is upper right
                    p = patches.Rectangle(
                        (left, bottom),
                        width,
                        height,
                        fill=False,
                        transform=ax.transAxes,
                        clip_on=False,
                    )
                    ax.text(
                        0.5 * (left + right),
                        0.5 * (bottom + top),
                        f"PC{i+1}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=20,
                        color=self.text_color,
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                a += 1

        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=self.text_color,
        )
        fig.tight_layout()
        matplotlib_savefig(
            savefig=savefig,
            DPI=DPI,
            output=output,
            name=name,
            fig_format=fig_format,
            transparent_background=transparent_background,
        )

    def save(self, output="output", name=""):
        directory_creator(output)
        # imposto la modalità di scrittura per scrivere i dati
        writer = pd.ExcelWriter(
            f"./{output}/{when_date()}_{name}PCA.xlsx", engine="xlsxwriter"
        )

        pd.DataFrame(self.correlation_matrix).to_excel(
            writer, sheet_name="Correlation Matrix"
        )
        pd.DataFrame(self.V).to_excel(writer, sheet_name="Eigenvalues")
        pd.DataFrame(self.L).to_excel(writer, sheet_name="Loadings Matrix")
        pd.DataFrame(self.L_r).to_excel(writer, sheet_name="Reduced Loadings Matrix")
        pd.DataFrame(self.PC_index).to_excel(writer, sheet_name="PCI Matrix")
        pd.DataFrame(self.T_r).to_excel(writer, sheet_name="Reduced Score Matrix")
        pd.DataFrame(self.variable_color).to_excel(writer, sheet_name="Variable Color")
        pd.DataFrame(self.sample_color).to_excel(writer, sheet_name="Sample Color")

        # writer.save()
        # writer.close()
        return print(f"File saved in: ./{output}/{when_date()}_{name}PCA.xlsx")
