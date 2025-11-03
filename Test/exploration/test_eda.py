"""
Unit tests for the ExploratoryDataAnalysis class.
"""

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.exploration.EDA import ExploratoryDataAnalysis

class TestEDA(unittest.TestCase):
    """
    Test suite for ExploratoryDataAnalysis.
    """

    def setUp(self):
        """Set up a sample mixed dataset for testing."""
        self.data = pd.DataFrame({
            'numerical': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'categorical': ['A', 'A', 'B', 'B', 'A', 'C', 'C', 'B', 'A', 'C']
        })
        self.eda = ExploratoryDataAnalysis(self.data)

    def test_univariate_summary(self):
        """Test the get_univariate_summary method."""
        summary = self.eda.get_univariate_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(summary.shape[1], 2) # 2 columns in the summary

    def test_plot_histogram(self):
        """Test the plot_histogram method."""
        fig = self.eda.plot_histogram('numerical')
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)  # Close the figure to avoid displaying it during tests

    def test_plot_boxplot(self):
        """Test the plot_boxplot method."""
        fig = self.eda.plot_boxplot('numerical')
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_scatter(self):
        """Test the plot_scatter method."""
        # Add another numerical column for scatter plot
        self.data['numerical2'] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        eda = ExploratoryDataAnalysis(self.data)
        fig = eda.plot_scatter('numerical', 'numerical2')
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_heatmap(self):
        fig = self.eda.plot_heatmap()
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_barchart(self):
        """Test the plot_barchart method."""
        fig = self.eda.plot_barchart('categorical')
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_run_chart(self):
        """Test the plot_run_chart method."""
        self.data['time'] = pd.to_datetime(pd.date_range('2023-01-01', periods=10, freq='D'))
        eda = ExploratoryDataAnalysis(self.data)
        fig = eda.plot_run_chart('time', 'numerical')
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_stem_and_leaf(self):
        """Test the plot_stem_and_leaf method."""
        import io
        from contextlib import redirect_stdout

        with io.StringIO() as buf, redirect_stdout(buf):
            self.eda.plot_stem_and_leaf('numerical')
            output = buf.getvalue()
        
        self.assertIn("Stem-and-leaf plot for numerical", output)
        self.assertIn("Stem | Leaf", output)

    def test_plot_scatter_3d(self):
        """Test the plot_scatter_3d method."""
        self.data['numerical2'] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        self.data['numerical3'] = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
        eda = ExploratoryDataAnalysis(self.data)
        fig = eda.plot_scatter_3d('numerical', 'numerical2', 'numerical3')
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_parallel_coordinates(self):
        """Test the plot_parallel_coordinates method."""
        fig = self.eda.plot_parallel_coordinates('categorical')
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

if __name__ == '__main__':
    unittest.main()
