"""
Unit tests for the FactorAnalysisOfMixedData class.
"""

import unittest
import pandas as pd
import numpy as np
from chemtools.dimensional_reduction.FactorAnalysisForMixedData import FactorAnalysisOfMixedData

class TestFAMD(unittest.TestCase):
    """
    Test suite for FactorAnalysisOfMixedData.
    """

    def setUp(self):
        """Set up a sample mixed dataset for testing."""
        self.data = pd.DataFrame({
            'quant1': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1],
            'quant2': [10.1, 9.0, 8.9, 7.8, 6.7, 5.6, 4.5, 3.4, 2.3, 1.2],
            'qual1': ['A', 'A', 'B', 'B', 'A', 'C', 'C', 'B', 'A', 'C'],
            'qual2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']
        })
        self.qualitative_variables = ['qual1', 'qual2']
        self.n_components = 2

    def test_famd_fit(self):
        """Test the fit method of FAMD."""
        famd = FactorAnalysisOfMixedData(n_components=self.n_components)
        famd.fit(self.data, qualitative_variables=self.qualitative_variables)

        # Check that the main attributes are created
        self.assertIsNotNone(famd.eigenvalues)
        self.assertIsNotNone(famd.T)  # Scores
        self.assertIsNotNone(famd.W)  # Loadings

        # Check the dimensions of the results
        n_objects = self.data.shape[0]
        self.assertEqual(famd.T.shape, (n_objects, self.n_components))
        
        # The number of loadings rows should correspond to the number of 
        # quantitative variables + the number of one-hot encoded qualitative variables.
        n_quant = len(famd.quantitative_variables)
        n_qual_dummies = famd._one_hot_encoder.get_feature_names_out().shape[0]
        total_vars_processed = n_quant + n_qual_dummies
        self.assertEqual(famd.W.shape, (total_vars_processed, self.n_components))

    def test_famd_transform(self):
        """Test the transform method of FAMD."""
        famd = FactorAnalysisOfMixedData(n_components=self.n_components)
        famd.fit(self.data, qualitative_variables=self.qualitative_variables)

        # Create new data for transformation
        new_data = pd.DataFrame({
            'quant1': [2.5, 7.5],
            'quant2': [8.5, 3.5],
            'qual1': ['B', 'C'],
            'qual2': ['X', 'Y']
        })

        transformed_data = famd.transform(new_data)
        self.assertEqual(transformed_data.shape, (2, self.n_components))

if __name__ == '__main__':
    unittest.main()
