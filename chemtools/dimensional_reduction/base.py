from chemtools.base import BaseModel
from abc import abstractmethod
import numpy as np


class DimensionalityReduction(BaseModel):
    """
    Base class for all dimensionality reduction methods in chemtools.

    This abstract class defines the interface for dimensionality reduction algorithms
    and inherits from BaseModel to maintain consistency across the library.

    Attributes:
        model_name (str): Name of the model
        method (str): Short identifier for the method
        n_components (int): Number of components to keep
    """

    def __init__(self, n_components: int = 2):
        super().__init__()
        self.n_components = n_components
        self.model_name = "Dimensionality Reduction"
        self.method = "DR"

    @abstractmethod
    def fit(self, X, variables_names=None, objects_names=None):
        """
        Fit the model to the data.

        Args:
            X (ndarray): Input data matrix
            variables_names (list, optional): Names of variables. Defaults to None.
            objects_names (list, optional): Names of objects. Defaults to None.
        """
        pass

    @abstractmethod
    def transform(self, X):
        """
        Apply dimensionality reduction to new data.

        Args:
            X (ndarray): New data to transform

        Returns:
            ndarray: Transformed data
        """
        pass

    def fit_transform(self, X, variables_names=None, objects_names=None):
        """
        Fit the model and transform the data in one step.

        Args:
            X (ndarray): Input data matrix
            variables_names (list, optional): Names of variables. Defaults to None.
            objects_names (list, optional): Names of objects. Defaults to None.

        Returns:
            ndarray: Transformed data
        """
        self.fit(X, variables_names, objects_names)
        return self.transform(X)