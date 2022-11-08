import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from decimal import Decimal
from tabulate import tabulate

from chemtools.regression import confidence_band
from chemtools.regression import prediction_band
from chemtools.utility import t_students
from chemtools.utility import when_date
from chemtools.utility import directory_creator
from chemtools.utility import array_to_column


class ordinary_least_squares:
    """Ordinay Least Squares class"""

    def __init__(self, x, y, alpha=0.05, name="test"):
        """_summary_

        Args:
            x (array, numpy array): indipendent variable
            y (array, numpy array): dipendent variable
            alpha (float, optional): level of confidence for student's t and confidence/prediction interval. Defaults to 0.05.
            name (str, optional): name of you ols object. Defaults to "test".

        Returns:
            self.x: indipendent variable
            self.y: dipendent variable
            self.name: name of you ols object
            self.alpha: level of confidence for student's t and confidence/prediction interval
            self.object_number: number of samples
            self.object_order: order of samples
            self.x_mean: mean of x
            self.y_mean: mean of y
            self.SSxx:
        """
        self.alpha = alpha
        self.name = name
        self.x = x
        self.y = y
        self.objects_number = x.shape[0]
        self.object_order = np.arange(1, x.shape[0] + 1)
        # Avarage of x and y
        self.x_mean = np.mean(x)
        self.y_mean = np.mean(y)
        # -SSxx
        self.SSxx = np.sum((x - self.x_mean) ** 2)
        # -SSyy
        self.SSyy = np.sum((y - self.y_mean) ** 2)
        # -SSxy
        self.SSxy = np.sum((x - self.x_mean) * (y - self.y_mean))
        # -S2x
        self.S2x = np.sum(x**2)
        # -S2x
        self.S2y = np.sum(y**2)
        # Slope
        self.slope = self.SSxy / self.SSxx
        # Intercept
        self.intercept = self.y_mean - self.x_mean * self.slope
        #####new x################################
        self.x_new = np.linspace(np.amin(x), np.amax(x), num=self.objects_number)
        #####Prediction of y (y pred)#############################################
        self.y_pred = np.array(self.x_new * self.slope + self.intercept)
        #####Prediction of y(x_orig) (y pred_orig)#############################################
        self.y_pred_orig = x * self.slope + self.intercept

        #######Standard Deviation#################
        #####sd_yx (standard deviation of y/x) Statistc y/x#############
        self.sd_yx = math.sqrt(
            (np.sum(y - self.y_pred_orig) ** 2) / (self.objects_number - 2)
        )
        #######Standard Deviation of slope##############
        self.sd_slope = self.sd_yx / math.sqrt(self.SSxx)
        #######Standard Deviation of intercept##############
        self.sd_intercept = self.sd_yx * (
            math.sqrt(np.sum(x**2) / (self.objects_number * self.SSxx))
        )
        #####SSres (RSS è la devianza residua (Residual Sum of Squares o ssr))####################################################################
        SSres = 0
        for i in range(self.objects_number):
            SSres += pow(y[i] - self.y_pred_orig[i], 2)
        self.SSres = SSres
        #####SStot(TSS è la devianza totale (Total Sum of Squares o sst))##########################
        SStot = 0
        for i in range(self.objects_number):
            SStot += pow(y[i] - self.y_mean, 2)
        self.SStot = SStot
        #####SSexp(ESS è la devianza spiegata dal modello (Explained Sum of Squares))##########################
        SSexp = 0
        for i in range(self.objects_number):
            SSexp += pow(self.y_pred_orig[i] - self.y_mean, 2)
        self.SSexp = SSexp
        #####SE ##########################################################################
        self.se = math.sqrt(self.SSres / (self.objects_number - 2))
        #####-Se-###########
        self.se2 = (
            1
            / (self.objects_number * (self.objects_number - 2))
            * (
                self.objects_number * self.SSyy
                - self.S2y
                - self.slope**2 * (self.objects_number * self.SSxx - self.S2x)
            )
        )
        #####-Sslope-###########
        self.Sslope2 = (
            self.objects_number
            * self.se2
            / (self.objects_number * self.SSxx - self.S2x)
        )
        # print(math.sqrt(Sslope2))
        #####-Sintercept-###########
        self.Sintercept2 = self.Sslope2 * (1 / self.objects_number) * self.SSxx
        # print(f"Sintercept:{math.sqrt(Sintercept2)}")

        #####t-score ######################################################################################
        d_f_ = self.objects_number - 2
        self.t_one, self.t_two = t_students(alpha, d_f_)

        #####Residual###################################################################################
        self.residual = y - self.y_pred_orig
        #####-Residual Standard Error-###################################################################################
        k = 1  # oltre all'intercetta ho anche un coefficiente angolare, se avessi solo l'intercetta satebbe 0
        # due è il numero di parametri che determino slope and intercept
        self.rse = math.sqrt(self.SSres / (self.objects_number - (k + 1)))

        ########## MSE (Mean Squared Error) ######################################
        def predict(x, b0, b1):
            return b0 + b1 * x

        mse = np.sum((y - predict(x, self.intercept, self.slope)) ** 2) / len(y)
        self.mse = mse
        # RMSE (Root Mean Squared Error)
        self.rmse = mse ** (1 / 2)
        #####R^2 (coefficiente di determinazione R2=SSexp/SStot or R2=1-(SSres/SStot))##################################################################
        self.R2 = self.SSexp / self.SStot
        self.r2 = 1 - (self.SSres / self.SStot)
        self.adjusted_r_squared = (
            1
            - ((self.objects_number - 1) / (self.objects_number - k - 1))
            * self.SSres
            / self.SStot
        )
        self.correlation_coefficent = f"{self.R2}\n{self.r2}\n{self.adjusted_r_squared}"

        #####Confidence interval########################################################################
        self.CI_Y_upper, self.CI_Y_lower = confidence_band.confidence_band(
            number_data=self.objects_number,
            x=x,
            x_mean=self.x_mean,
            y_pred_orig=self.y_pred_orig,
            SSxx=self.SSxx,
            t_two=self.t_two,
        )
        #####Prediction interval###################################################################
        self.PI_Y_upper, self.PI_Y_lower = prediction_band.prediction_band(
            number_data=self.objects_number,
            x=x,
            x_mean=self.x_mean,
            y_pred_orig=self.y_pred_orig,
            SSxx=self.SSxx,
            t_two=self.t_two,
        )
    def predict(self,x=None,y=None, decimal=1):
        """Dato un valore di y (o di x) da in dietro il valore di x (o di y) calcolato mediante l'equazione di regressione.

        Args:
            x (number): x value
            y (number): y value

        Returns:
            number: Valore di x (o di y) calcolato mediante l'equazione di regressione.
        """
        if y!=None:
            return round((y-self.intercept)/self.slope,decimal)
        elif x!=None:
            return round(self.slope*x+self.intercept, decimal)

    def scatter_plot(
        self,
        figursize=(6.4, 4.8),
        label="data",
        x_ax_label="x label",
        y_ax_label="y label",
        color_data="blue",
        marker_data="1",
        marker_size=2,
        savefig=False,
        fig_format="png",
        DPI=300,
        transparent_background=True,
        output="output",
        name="Data",
    ):
        fig, ax = plt.subplots(figsize=figursize)
        ax.scatter(
            self.x,
            self.y,
            label=label,
            color=color_data,
            marker=marker_data,
            s=marker_size,
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel(x_ax_label)
        ax.set_ylabel(y_ax_label)
        ax.legend()
        fig.tight_layout()
        if savefig == True:
            directory_creator(output)
            if DPI is None:
                plt.savefig(
                    f"./{output}/{when_date()}_{name}_plot.{fig_format}",
                    transparent=transparent_background,
                    bbox_inches="tight",
                )
            if DPI != None:
                plt.savefig(
                    f"./{output}/{when_date()}_{name}_plot.{fig_format}",
                    transparent=transparent_background,
                    bbox_inches="tight",
                    dpi=DPI,
                )
        return plt.show()

    def regression_plot(
        self,
        figursize=(6.4, 4.8),
        label="data",
        label2="line",
        x_ax_label="x label",
        y_ax_label="y label",
        line_color="black",
        color_data="blue",
        marker_data="1",
        marker_size=2,
        parameter_of_regression=True,
        scientific_notation=True,
        significant_figures_curve=4,
        significant_figures_r=4,
        savefig=False,
        fig_format="png",
        DPI=300,
        transparent_background=True,
        output="output",
        name="Regression line with data",
    ):
        fig, ax = plt.subplots(figsize=figursize)
        ax.plot(self.x_new, self.y_pred, label=label2, color=line_color)
        if parameter_of_regression == True:  # plot
            # create a text for plotting in legend al curve parameters
            mathematical_operator = "+" if self.intercept > 0 else ""
            if scientific_notation == True:  # use scientific notation for output
                textstr = "\n".join(
                    (
                        rf"$y= %.{significant_figures_curve}E\cdot x %s %.{significant_figures_curve}E$"
                        % (
                            Decimal(self.slope),
                            mathematical_operator,
                            Decimal(self.intercept),
                        ),
                        rf"$R=%.{significant_figures_r}f$" % (self.adjusted_r_squared,),
                    )
                )
            else:
                textstr = "\n".join(
                    (
                        rf"$y=%.{significant_figures_curve}f\cdot x %s %.{significant_figures_curve}f$"
                        % (self.slope, mathematical_operator, self.intercept),
                        rf"$R=%.{significant_figures_r}f$" % (self.adjusted_r_squared,),
                    )
                )
            ax.plot([], [], " ", label=textstr)
        ax.scatter(
            self.x,
            self.y,
            label=label,
            color=color_data,
            marker=marker_data,
            s=marker_size,
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel(x_ax_label)
        ax.set_ylabel(y_ax_label)
        ax.legend()
        fig.tight_layout()
        if savefig == True:
            directory_creator(output)
            if DPI is None:
                plt.savefig(
                    f"./{output}/{when_date()}_{name}_plot.{fig_format}",
                    transparent=transparent_background,
                    bbox_inches="tight",
                )
            if DPI != None:
                plt.savefig(
                    f"./{output}/{when_date()}_{name}_plot.{fig_format}",
                    transparent=transparent_background,
                    bbox_inches="tight",
                    dpi=DPI,
                )
        return plt.show()

    def residual_plot(
        self,
        figursize=(6.4, 4.8),
        label="Residual",
        x_ax_label="Number of Standard",
        y_ax_label="y_pred-y",
        title="Residual",
        color_line="black",
        color_data="red",
        marker_residual="o",
        marker_size=20,
        savefig=False,
        fig_format="png",
        DPI=300,
        transparent_background=True,
        output="output",
        name="Residual",
    ):
        fig, ax = plt.subplots(figsize=figursize)
        ax.scatter(
            self.object_order,
            self.residual,
            label=label,
            color=color_data,
            marker=marker_residual,
            s=marker_size,
        )
        ax.axhline(y=0, color=color_line)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(title)
        ax.set_xlabel(x_ax_label)
        ax.set_ylabel(y_ax_label)
        ax.legend()
        fig.tight_layout()
        if savefig == True:
            directory_creator(output)
            if DPI is None:
                plt.savefig(
                    f"./{output}/{when_date()}_{name}_plot.{fig_format}",
                    transparent=transparent_background,
                    bbox_inches="tight",
                )
            if DPI != None:
                plt.savefig(
                    f"./{output}/{when_date()}_{name}_plot.{fig_format}",
                    transparent=transparent_background,
                    bbox_inches="tight",
                    dpi=DPI,
                )
        return plt.show()

    def confidence_band_plot(
        self,
        figursize=(6.4, 4.8),
        label="ols",
        x_ax_label="x label",
        y_ax_label="y label",
        title="Regression with confident intervall",
        line_color="black",
        color_CI="grey",
        alpha_color=0.3,
        parameter_of_regression=True,
        scientific_notation=True,
        significant_figures_curve=4,
        significant_figures_r=4,
        savefig=False,
        fig_format="png",
        DPI=300,
        transparent_background=True,
        output="output",
        name="Regression line with confidence intervall",
    ):
        fig, ax = plt.subplots(figsize=figursize)
        ax.plot(self.x_new, self.y_pred, label=label, color=line_color)
        if parameter_of_regression == True:  # plot
            # create a text for plotting in legend al curve parameters
            mathematical_operator = "+" if self.intercept > 0 else ""
            if scientific_notation == True:  # use scientific notation for output
                textstr = "\n".join(
                    (
                        rf"$y= %.{significant_figures_curve}E\cdot x %s %.{significant_figures_curve}E$"
                        % (
                            Decimal(self.slope),
                            mathematical_operator,
                            Decimal(self.intercept),
                        ),
                        rf"$R=%.{significant_figures_r}f$" % (self.adjusted_r_squared,),
                    )
                )
            else:
                textstr = "\n".join(
                    (
                        rf"$y=%.{significant_figures_curve}f\cdot x %s %.{significant_figures_curve}f$"
                        % (self.slope, mathematical_operator, self.intercept),
                        rf"$R=%.{significant_figures_r}f$" % (self.adjusted_r_squared,),
                    )
                )
            ax.plot([], [], " ", label=textstr)
        ax.plot(self.x, self.CI_Y_upper, color=color_CI)
        ax.plot(self.x, self.CI_Y_lower, color=color_CI)
        ax.fill_between(
            self.x,
            self.CI_Y_upper,
            self.CI_Y_lower,
            color=color_CI,
            alpha=alpha_color,
            label=f"Confident intervall at {(1-self.alpha)*100}%",
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(title)
        ax.set_xlabel(x_ax_label)
        ax.set_ylabel(y_ax_label)
        ax.legend()
        fig.tight_layout()
        if savefig == True:
            directory_creator(output)
            if DPI is None:
                plt.savefig(
                    f"./{output}/{when_date()}_{name}_plot.{fig_format}",
                    transparent=transparent_background,
                    bbox_inches="tight",
                )
            if DPI != None:
                plt.savefig(
                    f"./{output}/{when_date()}_{name}_plot.{fig_format}",
                    transparent=transparent_background,
                    bbox_inches="tight",
                    dpi=DPI,
                )
        return plt.show()

    def prediction_band_plot(
        self,
        figursize=(6.4, 4.8),
        label="ols",
        x_ax_label="x label",
        y_ax_label="y label",
        title="Regression line with prediction intervall",
        line_color="black",
        color_CI="lightgrey",
        alpha_color=0.3,
        parameter_of_regression=True,
        scientific_notation=True,
        significant_figures_curve=4,
        significant_figures_r=4,
        savefig=False,
        fig_format="png",
        DPI=300,
        transparent_background=True,
        output="output",
        name="Regression line with prediction intervall",
    ):
        fig, ax = plt.subplots(figsize=figursize)
        ax.plot(self.x_new, self.y_pred, label=label, color=line_color)
        if parameter_of_regression == True:  # plot
            # create a text for plotting in legend al curve parameters
            mathematical_operator = "+" if self.intercept > 0 else ""
            if scientific_notation == True:  # use scientific notation for output
                textstr = "\n".join(
                    (
                        rf"$y= %.{significant_figures_curve}E\cdot x %s %.{significant_figures_curve}E$"
                        % (
                            Decimal(self.slope),
                            mathematical_operator,
                            Decimal(self.intercept),
                        ),
                        rf"$R=%.{significant_figures_r}f$" % (self.adjusted_r_squared,),
                    )
                )
            else:
                textstr = "\n".join(
                    (
                        rf"$y=%.{significant_figures_curve}f\cdot x %s %.{significant_figures_curve}f$"
                        % (self.slope, mathematical_operator, self.intercept),
                        rf"$R=%.{significant_figures_r}f$" % (self.adjusted_r_squared,),
                    )
                )
            ax.plot([], [], " ", label=textstr)
        ax.plot(self.x, self.PI_Y_upper, color=color_CI)
        ax.plot(self.x, self.PI_Y_lower, color=color_CI)
        ax.fill_between(
            self.x,
            self.PI_Y_upper,
            self.PI_Y_lower,
            color=color_CI,
            alpha=alpha_color,
            label=f"Prediction intervall at {(1-self.alpha)*100}%",
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(title)
        ax.set_xlabel(x_ax_label)
        ax.set_ylabel(y_ax_label)
        ax.legend()
        fig.tight_layout()
        if savefig == True:
            directory_creator(output)
            if DPI is None:
                plt.savefig(
                    f"./{output}/{when_date()}_{name}_plot.{fig_format}",
                    transparent=transparent_background,
                    bbox_inches="tight",
                )
            if DPI != None:
                plt.savefig(
                    f"./{output}/{when_date()}_{name}_plot.{fig_format}",
                    transparent=transparent_background,
                    bbox_inches="tight",
                    dpi=DPI,
                )
        return plt.show()

    def summary(self):
        headers = ["Number of sample", "Avarage of x", "Avarage of y"]
        table = [[self.objects_number, self.x_mean, self.y_mean]]

        headers1 = [
            "SSxx",
            "SSxy",
            "SSres",
            "SStot",
            "SSexp",
            "MSE\n(Mean Squared Error)",
        ]
        table1 = [[self.SSxx, self.SSxy, self.SSres, self.SStot, self.SSexp, self.mse]]

        headers2 = [
            "RMSE\n(Root Mean Squared Error)",
            "R^2=SSexp/SStot\nR^2=1-(SSres/SStot)\nadjusted R^2",
        ]
        table2 = [[self.rmse, self.correlation_coefficent]]

        headers3 = ["SE", "α", "tα/2", "tα"]
        table3 = [[self.se, self.alpha, self.t_two, self.t_one]]

        headers4 = ["Slope\nb1", "Intercept\nb0"]
        table4 = [[self.slope, self.intercept]]

        headers5 = ["X value", "Y value", "Predicted Y for original X", "Residual"]
        table5 = [
            [
                array_to_column(self.x),
                array_to_column(self.y),
                array_to_column(self.y_pred_orig),
                array_to_column(self.residual),
            ]
        ]

        headers6 = ["New value for X", "Y predicted"]
        table6 = [[self.x_new, self.y_pred]]

        headers7 = ["Confidence Interval lower", "Confidence Interval upper"]
        table7 = [[self.CI_Y_lower, self.CI_Y_upper]]

        headers8 = ["Prediction Interval lower", "Prediction Interval upper"]
        table8 = [[self.PI_Y_lower, self.PI_Y_upper]]

        print(f"Print of data calculated of {self.name}:")
        print("\n")
        print(tabulate(table, headers))
        print("\n")
        print(tabulate(table1, headers1))
        print("\n")
        print(tabulate(table2, headers2))
        print("\n")
        print(tabulate(table3, headers3))
        print("\n")
        print(tabulate(table4, headers4))
        print("\n")
        print(tabulate(table5, headers5))
        print("\n")
        print(tabulate(table6, headers6))
        print("\n")
        print(tabulate(table7, headers7))
        print("\n")
        print(tabulate(table8, headers8))
