"""Tests for PCA module."""

import pytest
import numpy as np
from chemtools.exploration import PrincipalComponentAnalysis


def test_pca_fit(sample_data, variable_names, object_names):
    """Test PCA fitting."""
    X, _ = sample_data
    pca = PrincipalComponentAnalysis()
    pca.fit(X, variables_names=variable_names, objects_names=object_names)

    assert hasattr(pca, "eigenvalues")
    assert hasattr(pca, "eigenvectors")
    assert pca.eigenvalues.shape[0] == X.shape[1]


def test_pca_transform(sample_data):
    """Test PCA transformation."""
    X, _ = sample_data
    pca = PrincipalComponentAnalysis()
    pca.fit(X)

    transformed = pca.transform(X)
    assert transformed.shape == X.shape


def test_pca_explained_variance(sample_data):
    """Test explained variance calculation."""
    X, _ = sample_data
    pca = PrincipalComponentAnalysis()
    pca.fit(X)

    explained_var = pca.explained_variance_ratio
    assert np.all(explained_var >= 0)
    assert np.all(explained_var <= 1)
    np.testing.assert_almost_equal(np.sum(explained_var), 1.0)
