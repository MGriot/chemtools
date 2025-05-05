"""Tests for regression module."""

import pytest
import numpy as np
from chemtools.regression import LinearRegression


def test_linear_regression_fit(sample_data):
    """Test basic linear regression fitting."""
    X, y = sample_data
    model = LinearRegression()
    model.fit(X, y)

    assert hasattr(model, "coefficients")
    assert model.coefficients.shape[0] == X.shape[1] + 1  # +1 for intercept


def test_linear_regression_predict(sample_data):
    """Test prediction functionality."""
    X, y = sample_data
    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == y.shape


def test_linear_regression_summary(sample_data):
    """Test summary generation."""
    X, y = sample_data
    model = LinearRegression()
    model.fit(X, y)

    summary = model.summary
    assert isinstance(summary, str)
    assert "Linear Regression Summary" in summary
