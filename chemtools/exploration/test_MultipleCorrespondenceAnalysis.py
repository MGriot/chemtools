import unittest
import numpy as np
from MultipleCorrespondenceAnalysis import (
    MultipleCorrespondenceAnalysis,
)


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
        self.assertEqual(self.mca.n_variables, 3)
        self.assertEqual(self.mca.n_objects, 3)
        self.assertEqual(len(self.mca.variables_colors), 3)
        self.assertEqual(len(self.mca.objects_colors), 3)
        self.assertEqual(self.mca.correspondence_matrix.shape, (3, 3))
        self.assertEqual(self.mca.row_profiles.shape, (3, 3))
        self.assertEqual(self.mca.col_profiles.shape, (3, 3))
        self.assertEqual(self.mca.V.shape, (3,))
        self.assertEqual(self.mca.L.shape, (3, 3))

    def test_change_variables_colors(self):
        colors = self.mca.change_variables_colors()
        self.assertEqual(len(colors), self.mca.n_variables)

    def test_change_objects_colors(self):
        colors = self.mca.change_objects_colors()
        self.assertEqual(len(colors), self.mca.n_objects)


if __name__ == "__main__":
    unittest.main()
