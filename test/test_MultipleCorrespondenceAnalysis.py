import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from chemtools.exploration.MultipleCorrespondenceAnalysis import (
    MultipleCorrespondenceAnalysis,
)
from chemtools.plots.exploration import mca_plots


class TestMultipleCorrespondenceAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Example contingency table
        cls.X = np.array(
            [
                [10, 5, 15],
                [5, 15, 10],
                [15, 10, 5],
            ]
        )
        cls.variables_names = ["A", "B", "C"]
        cls.objects_names = ["X", "Y", "Z"]
        cls.mca = MultipleCorrespondenceAnalysis()
        cls.mca.fit(cls.X, cls.variables_names, cls.objects_names)

    def test_fit(self):
        print("Eigenvalues (V):", self.mca.V)
        print("Object Coordinates (L):", self.mca.L)
        print("Variable Coordinates (G):", self.mca.G)

        # Add assertions based on expected values
        np.testing.assert_array_almost_equal(
            self.mca.V, [0.22222222, 0.11111111, 0.0], decimal=4
        )
        # Add more assertions as needed

    def test_plots(self):
        # Call the plotting functions, but suppress the actual plot output
        mca_plots.plot_eigenvalues(self.mca)
        mca_plots.plot_objects(self.mca)
        mca_plots.plot_variables(self.mca)


if __name__ == "__main__":
    unittest.main()
