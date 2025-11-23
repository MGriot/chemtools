"""Tests for plotting system."""

import pytest
from chemtools.plots.base import BasePlotter


def test_plotter_initialization():
    """Test basic plotter initialization."""
    plotter = BasePlotter()
    assert plotter.library == "matplotlib"  # default
    assert plotter.theme == "classic_professional_light"  # default from BasePlotter
    assert plotter.style_preset == "default" # default


def test_plotter_theme_initialization():
    """Test theme initialization."""
    # Use an existing dark theme
    plotter = BasePlotter(theme="classic_professional_dark")
    assert plotter.theme == "classic_professional_dark"
    # Assuming dark theme bg color is "#2b2b2b" from previous check or actual theme file
    assert plotter.colors["bg_color"] == "#2b2b2b"

    # Use an existing light theme
    plotter_light = BasePlotter(theme="classic_professional_light")
    assert plotter_light.theme == "classic_professional_light"
    # Add more assertions for light theme colors if known
    # For example, if light theme bg color is known:
    # assert plotter_light.colors["bg_color"] == "#f0f0f0"


def test_plotter_style_preset_initialization():
    """Test style preset initialization."""
    plotter = BasePlotter(style_preset="minimal")
    assert plotter.style_preset == "minimal"
    # Add assertions for specific style preset settings


def test_plotter_backend_initialization():
    """Test plotting backend initialization."""
    plotter = BasePlotter(library="plotly")
    assert plotter.library == "plotly"

    plotter_mpl = BasePlotter(library="matplotlib")
    assert plotter_mpl.library == "matplotlib"