import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from chemtools.dimensional_reduction.dimension_reduction import DimensionalityReduction


class TestDimensionalityReduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Example data
        cls.X = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ]
        )
        cls.variables_names = ["A", "B", "C"]
        cls.objects_names = ["X1", "X2", "X3", "X4"]

    def test_fit(self):
        model = DimensionalityReduction()
        with self.assertRaises(NotImplementedError):
            model.fit(self.X)

    def test_transform(self):
        model = DimensionalityReduction()
        with self.assertRaises(NotImplementedError):
            model.transform(self.X)

    def test_fit_transform(self):
        model = DimensionalityReduction()
        with self.assertRaises(NotImplementedError):
            model.fit_transform(self.X)

    def test_score(self):
        model = DimensionalityReduction()
        with self.assertRaises(NotImplementedError):
            model.score(self.X)


if __name__ == "__main__":
    unittest.main()
