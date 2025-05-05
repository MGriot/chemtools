"""Tests for preprocessing module."""

import pytest
import numpy as np
from chemtools.preprocessing import (
    autoscaling,
    matrix_mean,
    matrix_standard_deviation,
    correlation_matrix,
)


def test_autoscaling():
    """Test autoscaling functionality."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_scaled = autoscaling(X)
    assert X_scaled.shape == X.shape
    np.testing.assert_almost_equal(X_scaled.mean(axis=0), [0, 0])
    np.testing.assert_almost_equal(X_scaled.std(axis=0), [1, 1])


def test_matrix_mean():
    """Test matrix mean calculation."""
    X = np.array([[1, 2], [3, 4]])
    means = matrix_mean(X)
    np.testing.assert_array_equal(means, [2, 3])


def test_correlation_matrix():
    """Test correlation matrix calculation."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    corr = correlation_matrix(X)
    assert corr.shape == (2, 2)
    np.testing.assert_almost_equal(np.diag(corr), [1, 1])
