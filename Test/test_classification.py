import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from chemtools.classification.PrincipalComponentClassificationAnalysis import (
    PrincipalComponentClassificationAnalysis,
)


class TestPrincipalComponentClassificationAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Example data
        cls.X = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
            ]
        )
        cls.y = np.array([0, 0, 0, 1, 1, 1])
        cls.variables_names = ["A", "B", "C"]
        cls.objects_names = ["X1", "X2", "X3", "X4", "X5", "X6"]
        cls.pcca = PrincipalComponentClassificationAnalysis()

    def test_fit(self):
        self.pcca.fit(self.X, self.y, self.variables_names, self.objects_names)
        # Add assertions to check the fitted model attributes
        # For example:
        # self.assertEqual(self.pcca.n_components, 2)  # Check number of components
        # np.testing.assert_array_almost_equal(
        #     self.pcca.explained_variance_ratio_, [0.9, 0.1], decimal=1
        # )  # Check explained variance

    def test_predict(self):
        self.pcca.fit(self.X, self.y, self.variables_names, self.objects_names)
        X_new = np.array([[2, 3, 4], [8, 9, 10]])
        y_pred = self.pcca.predict(X_new)
        # Add assertions to check the predicted values
        # For example:
        # np.testing.assert_array_equal(y_pred, [0, 1])

    # Add more test methods as needed


if __name__ == "__main__":
    unittest.main()
