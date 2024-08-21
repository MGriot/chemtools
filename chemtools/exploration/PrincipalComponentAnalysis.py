# ---------------------------------------------------------
# Author: MGriot
# Date: 21/08/2024
#
# This code is protected by copyright and cannot be 
# copied, modified, or used without the explicit 
# permission of the author. All rights reserved.
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import f, norm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from chemtools.preprocessing import autoscaling
from chemtools.preprocessing.matrix_standard_deviation import matrix_standard_deviation
from chemtools.preprocessing import correlation_matrix
from chemtools.preprocessing import diagonalized_matrix
from chemtools.utility import reorder_array
from chemtools.utility import heatmap
from chemtools.utility import annotate_heatmap
from chemtools.utility.set_names import set_objects_names, set_variables_names
from chemtools.utility import random_colorHEX

class PrincipalComponentAnalysis:
    def __init__(self,X, variables_names=None, objects_names=None):
        self.X=X
        self.variables=set_variables_names(self.X,variables_names)
        self.objects=set_objects_names(self.X,objects_names)
        self.n_variables=self.X.shape[1]
        self.n_objects=self.X.shape[0]
        self.variables_colors=self.change_variables_colors()
        self.objects_colors=self.change_objects_colors()
    
    def fit(self):
        # Calcola la media
        self.mean = np.mean(self.X, axis=0)
        #calcola le deviazioni standard della matrice
        self.std=matrix_standard_deviation(self.X, mode="column")
        # Calcola la matrice autoscalata
        self.X_autoscaled = autoscaling(self.X)
        #Calcola la matrice do covarianza
        self.correlation_matrix = correlation_matrix(self.X_autoscaled)
        # eighenvalue enighenvector
        self.V, self.L = diagonalized_matrix(self.correlation_matrix)
        self.V_ordered, self.order = reorder_array(self.V)
        
        # riordino le PCi
        PC_index = np.array([["PC{}".format(i+1) for i in range(self.V.shape[0])], self.order, self.V_ordered])
        self.PC_index = PC_index
        self.number_of_PC = self.L.shape[1]
        
        self.L_ordered=self.L[self.order]
        # Criteri scelta nPC
        self.num_eigenvalues_greater_than_one=np.argmax(pca.V_ordered < 1)
    
    def reduction(self, n_components):
        self.n_component = n_components
        self.V_reduced=self.V_ordered[:n_components]
        # Calcola la matrice di trasformazione
        self.W = self.L_ordered[:, :n_components]
        # Riduce la dimensionalità dei dati
        self.T = np.dot(self.X_autoscaled, self.W)
    
    def statistics(self,alpha=0.05):
        self.X_reconstructed = np.dot(self.T, self.W.T)
        self.E = self.X_autoscaled - self.X_reconstructed
        self.T2 = np.diag(self.T @ np.diag(self.V_ordered[:self.n_component] ** (-1)) @ self.T.T)
        self.T2con = self.T @ np.diag(self.V_ordered[:self.n_component] ** (-1 / 2)) @ self.W.T
        self.Q = np.sum(self.E ** 2, axis=1)
        self.Qcon = self.E
        self.T2_critical_value=self.hotellings_t2_critical_value(alpha=alpha)
    

    def hotellings_t2_critical_value(self, alpha=0.05):
        p = self.n_variables
        n = self.n_objects
        f_critical_value = f.ppf(1 - alpha, p, n - p)
        return (p * (n - 1)) / (n - p) * f_critical_value
    
    def change_variables_colors(self):
        return random_colorHEX(self.n_variables)
    def change_objects_colors(self):
        return random_colorHEX(self.n_objects)

    ### PLots ----------------------------------------------------------------
    def plot_correlation_matrix(self, cmap="coolwarm", threshold=None):
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
        im, cbar = heatmap(self.correlation_matrix, self.variables, self.variables, ax=ax, cmap=cmap, cbarlabel="Correlation value")
        texts = annotate_heatmap(im, threshold=threshold, valfmt="{x:.2f}")
        ax.set_title("Correlation Matrix")
        plt.tight_layout()

    def plot_eigenvalues_greater_than_one(self):
        #plt.figure(figsize=(self.PC_index.shape[1], 5))
        # grafico degli autovalori maggiori di 1
        # da verificare un attimo se non rimane uguale all'autovalore medio
        plt.axvline(
            x=self.num_eigenvalues_greater_than_one-0.5, #lìultimo 1 è la condizione
            color="brown",
            linestyle="-",
            label="Autovalori maggiori di 1",
        )
        plt.xticks(range(self.PC_index.shape[1]), range(1,self.PC_index.shape[1]+1))
        plt.xlabel(r"$PC_i$")
        plt.legend(loc="best")
        plt.title("Eigenvalue")

    def plot_eigenvalues_variance(self):
        #plt.figure(figsize=(self.PC_index.shape[1], 5))
        # grafico degli autovalori
        plt.bar(
            x=self.PC_index[0, :],
            height=(self.V_ordered/ self.V_ordered.sum()) * 100,
            fill=False,
            edgecolor="darkorange",
            label="Varianza %",
        )
        plt.xticks(range(self.PC_index.shape[1]), range(1,self.PC_index.shape[1]+1))
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")

    def plot_cumulative_variance(self):
        c = np.cumsum(self.V_ordered / self.V_ordered.sum()) * 100
        plt.bar(
            x=self.PC_index[0, :],
            height=c,
            fill=False,
            edgecolor="black",
            linestyle="--",
            width=0.6,
            label="Varianza cumulata %",
        )
        plt.xticks(range(self.PC_index.shape[1]), range(1,self.PC_index.shape[1]+1))
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")

    def plot_average_eigenvalue_criterion(self):
        plt.axvline(
            x=np.argmax(self.V_ordered < self.V_ordered.mean()) - 0.5,
            color="red",
            alpha=0.5,
            linestyle="-",
            label="AEC",
        )
        plt.xticks(range(self.PC_index.shape[1]), range(1,self.PC_index.shape[1]+1))
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")

    def plot_KP_criterion(self):
        rank = np.linalg.matrix_rank(self.correlation_matrix)
        sum_term = sum(self.V[m] / self.V.sum() - 1 / self.V.size for m in range(rank))
        x = round(1+(self.V.size-1)*(1-((sum_term+(self.V.size-rank)**(1/self.V.size))/(2*(self.V.size-1)/self.V.size))))-1
        plt.axvline(
            x=x,
            color="purple",
            alpha=0.5,
            linestyle="--",
            label="KP",
        )
        plt.xticks(range(self.PC_index.shape[1]), range(1,self.PC_index.shape[1]+1))
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")

    def plot_KL_criterion(self):
        rank = np.linalg.matrix_rank(self.correlation_matrix)
        sum_term = sum(self.V[m] / self.V.sum() - 1 / self.V.size for m in range(rank))
        x = round(self.V.size**(1-(sum_term+(self.V.size-rank)**(1/self.V.size))/(2*(self.V.size-1)/self.V.size)))-1
        plt.axvline(
            x=x,
            color="cyan",
            alpha=0.5,
            linestyle="-",
            label="KL",
        )
        plt.xticks(range(self.PC_index.shape[1]), range(1,self.PC_index.shape[1]+1))
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")

    def plot_CAEC_criterion(self):
        plt.axvline(
            x=np.argmax(self.V_ordered < 0.7 * self.V_ordered.mean())-0.5,
            color="blue",
            alpha=0.5,
            linestyle="--",
            label="CAEC",
        )
        plt.xticks(range(self.PC_index.shape[1]), range(1,self.PC_index.shape[1]+1))
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")

    def plot_broken_stick(self):
        n = self.V_ordered.shape[0]
        dm = (100 / n) * np.cumsum(1 / np.arange(1, n + 1)[::-1])
        plt.plot(self.PC_index[0, :], dm, color="lightgreen", label="Broken stick")
        plt.xticks(range(self.PC_index.shape[1]), range(1,self.PC_index.shape[1]+1))
        plt.xlabel(r"$PC_i$")
        plt.ylabel(r"$\lambda$%")
        plt.legend(loc="best")
        plt.title("Eigenvalue")
    
    def plot_eigenvalue(self):
        self.plot_eigenvalues_greater_than_one()
        self.plot_eigenvalues_variance()
        self.plot_cumulative_variance()
        self.plot_average_eigenvalue_criterion()
        self.plot_KP_criterion()
        self.plot_KL_criterion()
        self.plot_CAEC_criterion()
        self.plot_broken_stick()

    def plot_hotteling_t2_vs_q(self):
        # Calcola Q critico
        #Q_critico =f_value/(self.T.shape[0]-self.T.shape[1])
        # Crea il grafico
        for i in range(len(self.Q)):
            plt.plot(self.Q[i], self.T2[i], 'o', label=self.objects[i])
        # Aggiungi il valore di T2 critico
        #plt.axhline(y=T2_critico, color='r', linestyle='-', label=r"$T^2_{crit}$")
        # Aggiungi il valore di Q critico
        #plt.axvline(x=Q_critico, color='r', linestyle='-', label=r"$Q_{crit}$")
        plt.xlabel(r'$Q$')
        plt.ylabel(r'$Hotteling\'s T^2$')
        plt.legend(loc="best")

    def plot_pci_contribution(self, text_color="black"):
        for i in range(self.W.shape[1]):
            plt.plot(np.arange(self.n_variables),self.W[:, i],marker="o",markerfacecolor="none",label=f"PC$_{i+1}$")
        plt.title(f"Contributions of the PC$_i$")
        plt.xticks(np.arange(self.n_variables), self.variables)
        plt.legend(labelcolor=text_color)
        plt.xlabel("Variable")
        plt.ylabel("Value of loading")
    

    def plot_loadings(self,arrows=True, text_color="black"):
        # Loadings plot
        fig, axs = plt.subplots(
            self.n_component,
            self.n_component,
            figsize=(5 * self.n_component, 5 * self.n_component),
        )
        a = 1
        for i in range(self.n_component):
            for j in range(self.n_component):
                fig.suptitle("Loadings plot", fontsize=24, y=1, color=text_color)
                ax = plt.subplot(self.n_component, self.n_component, a)
                if i != j:
                    for d in range(self.n_variables):
                        #######################
                        if self.W[d, i] > self.W[:, i].mean():
                            if self.W[d, j] > self.W[:, j].mean():
                                position = ["left", "bottom"]
                            elif self.W[d, j] < self.W[:, j].mean():
                                position = ["right", "top"]
                        elif self.W[d, i] < self.W[:, i].mean():
                            if self.W[d, j] > self.W[:, j].mean():
                                position = ["left", "bottom"]
                            elif self.W[d, j] < self.W[:, j].mean():
                                position = ["right", "top"]
                        ######################
                        ax.scatter(
                            x=self.W[d, i],
                            y=self.W[d, j],
                            label=self.variables[d],
                            color=self.variables_colors[d],
                        )
                        if arrows == True:
                            ax.arrow(
                                0,
                                0,
                                self.W[d, i],
                                self.W[d, j],
                                length_includes_head=True,
                                head_width=0.015,
                                color=text_color,
                                alpha=0.3,
                            )
                        ax.annotate(
                            text=self.variables[d],
                            xy=(self.W[d, i], self.W[d, j]),
                            ha=position[0],
                            va=position[1],
                            color=text_color,
                        )
                        ax.set_xlabel(rf"PC$_{i+1}$")
                        ax.set_ylabel(rf"PC$_{j+1}$")
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
                        rf"PC$_{i+1}$",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=20,
                        color=text_color,
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
            labelcolor=text_color,
        )

    def plot_scores(self,label_point=False,text_color="black"):
        # Scores plot
        fig, axs = plt.subplots(
            self.n_component,
            self.n_component,
            figsize=(5 * self.n_component, 5 * self.n_component),
        )
        a = 1
        for i in range(self.n_component):
            for j in range(self.n_component):
                fig.tight_layout()
                fig.suptitle("Scores plot", fontsize=24, y=1, color=text_color)
                ax = plt.subplot(self.n_component, self.n_component, a)
                if i != j:
                    for d in range(self.n_objects):
                        # ax.scatter(x=T_r[d,i], y=T_r[d,j], c=sample_color[d], edgecolors='black', label=sample[d])
                        ax.scatter(
                            x=self.T[d, i],
                            y=self.T[d, j],
                            c=self.objects_colors[d],
                            label=self.objects[d],
                        )
                        if label_point == True:
                            try:
                                ax.annotate(
                                    text=self.objects[d][0],
                                    xy=(self.T[d, i], self.T[d, j]),
                                    color=text_color,
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
                        color=text_color,
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
            labelcolor=text_color,
        )
    def plot_biplot(
            self,
            label_point=False,
            subplot_dimensions=[5, 5],text_color="black"):
            # biplots
            fig, axs = plt.subplots(
                self.n_component,
                self.n_component,
                figsize=(
                    subplot_dimensions[0] * self.n_component,
                    subplot_dimensions[1] * self.n_component,
                ),
            )
            fig.suptitle("Biplots plot", fontsize=24, y=1, color=text_color)
            a = 1
            for i in range(self.n_component):
                for j in range(self.n_component):
                    ax = plt.subplot(self.n_component, self.n_component, a)
                    if i != j:
                        for e in range(self.n_objects):
                            # ax.scatter(x=T_r[e,i], y=T_r[e,j], c=sample_color[e], edgecolors='black', label=sample[e])
                            ax.scatter(
                                x=self.T[e, i],
                                y=self.T[e, j],
                                c=self.objects_colors[e],
                                label=self.objects[e],
                            )
                            if label_point == True:
                                try:
                                    ax.annotate(
                                        text=self.objects[d],
                                        xy=(self.T[d, i], self.T[d, j]),
                                        color=text_color,
                                    )
                                except:
                                    pass
                        for d in range(self.n_variables):
                            #######################
                            if self.W[d, i] > self.W[:, i].mean():
                                if self.W[d, j] > self.W[:, j].mean():
                                    position = ["left", "bottom"]
                                elif self.W[d, j] < self.W[:, j].mean():
                                    position = ["right", "top"]
                            elif self.W[d, i] < self.W[:, i].mean():
                                if self.W[d, j] > self.W[:, j].mean():
                                    position = ["left", "bottom"]
                                elif self.W[d, j] < self.W[:, j].mean():
                                    position = ["right", "top"]
                            ######################
                            ax.arrow(
                                0,
                                0,
                                pow(self.T[:, i] ** 2, 0.5).max() * self.W[d, i],
                                pow(self.T[:, j] ** 2, 0.5).max() * self.W[d, j],
                                length_includes_head=True,
                                head_width=0.15,
                                color=text_color,
                                alpha=0.3,
                            )
                            ax.annotate(
                                text=self.variables[d],
                                xy=(
                                    pow(self.T[:, i] ** 2, 0.5).max() * self.W[d, i],
                                    pow(self.T[:, j] ** 2, 0.5).max() * self.W[d, j],
                                ),
                                ha=position[0],
                                va=position[1],
                                color=text_color,
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
                            color=text_color,
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
                labelcolor=text_color)
