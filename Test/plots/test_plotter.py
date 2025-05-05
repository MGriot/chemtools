"""Tests for plotting system."""

import pytest
from chemtools.plots import Plotter


def test_plotter_initialization():
    """Test basic plotter initialization."""
    plotter = Plotter()
    assert plotter.library == "matplotlib"  # default
    assert plotter.theme == "light"  # default


def test_plotter_theme_switching():
    """Test theme switching functionality."""
    plotter = Plotter()
    plotter.set_theme("dark")
    assert plotter.theme == "dark"
    assert plotter.colors["bg_color"] == "#2b2b2b"


def test_plotter_style_presets():
    """Test style preset application."""
    plotter = Plotter()
    plotter.set_style_preset("minimal")
    # Add assertions for style preset settings


def test_plotter_backend_switching():
    """Test plotting backend switching."""
    plotter = Plotter(library="plotly")
    assert plotter.library == "plotly"

    plotter.set_library("matplotlib")
    assert plotter.library == "matplotlib"
