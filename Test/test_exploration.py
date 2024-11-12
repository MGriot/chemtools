import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from chemtools.exploration.GeneralizedCanonicalCorrelationAnalysis import (
    GeneralizedCanonicalCorrelationAnalysis,
)


class TestGeneralizedCanonicalCorrelationAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Example data (replace with your actual data)
        cls.X1 = np.array([[1, 2], [3, 4], [5, 6]])
        cls.X2 = np.array([[7, 8], [9, 10], [11, 12]])
        cls.X3 = np.array([[13, 14], [15, 16], [17, 18]])

        cls.variables_names = ["Var1", "Var2"]
        cls.objects_names = ["Obj1", "Obj2", "Obj3"]

        cls.gcca = GeneralizedCanonicalCorrelationAnalysis()

    def test_fit(self):
        self.gcca.fit([self.X1, self.X2, self.X3], self.variables_names, self.objects_names)
        # Add assertions to check the fitted model attributes
        # For example:
        # self.assertEqual(self.gcca.n_components, 2)  # Check number of components
        # np.testing.assert_array_almost_equal(
        #     self.gcca.explained_variance_ratio_, [0.9, 0.1], decimal=1
        # )  # Check explained variance

    def test_transform(self):
        self.gcca.fit([self.X1, self.X2, self.X3], self.variables_names, self.objects_names)
        X_new = [
            np.array([[2, 3], [4, 5], [6, 7]]),
            np.array([[8, 9], [10, 11], [12, 13]]),
            np.array([[14, 15], [16, 17], [18, 19]]),
        ]
        transformed_data = self.gcca.transform(X_new)
        # Add assertions to check the transformed data
        # For example:
        # self.assertEqual(len(transformed_data), 3)  # Check number of datasets
        # self.assertEqual(transformed_data[0].shape, (3, 2))  # Check shape of transformed data

    # Add more test methods as needed


if __name__ == "__main__":
    unittest.main()
