"""Configuration and fixtures for pytest."""

import pytest
import numpy as np


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def variable_names():
    """Generate sample variable names."""
    return [f"Var_{i}" for i in range(5)]


@pytest.fixture
def object_names():
    """Generate sample object names."""
    return [f"Obj_{i}" for i in range(100)]
