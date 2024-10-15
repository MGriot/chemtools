import pickle
from abc import ABC, abstractmethod


class BaseModel(ABC):  # Make BaseModel an Abstract Base Class
    """
    Abstract base class for statistical models, providing common
    functionalities like saving/loading and a basic summary.
    """

    def __init__(self):
        self.model_name = "Base Model"  # Default name

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

    @abstractmethod
    def _get_summary_data(self):
        """
        Abstract method that should be implemented by subclasses
        to return a dictionary of data for the summary.
        """
        pass

    def summary(self):
        """Prints a formatted summary of the model."""
        summary_data = self._get_summary_data()

        # Common Summary Information
        print("-" * 30)
        print(f"{self.model_name} Summary")
        print("-" * 30)
        print(f"Model: {type(self).__name__}")
        # ... (Add any other general model information here)

        # Subclass-Specific Summary Data
        for key, value in summary_data.items():
            print(f"{key}: {value}")
        print("-" * 30)
