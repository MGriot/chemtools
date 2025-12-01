# Project Overview: Chemtools

## Introduction

`chemtools` is a Python library designed to provide a comprehensive suite of chemometrics tools for data analysis. It aims to simplify complex statistical and machine learning tasks commonly encountered in chemistry, chemical engineering, and related scientific disciplines. The library offers functionalities for data exploration, preprocessing, regression, classification, dimensional reduction, and advanced statistical analysis, all built upon a consistent and extensible architecture.

## Core Architectural Principles

The `chemtools` library is built around two fundamental abstract base classes: `BaseModel` and `BasePlotter`. These classes establish a unified interface and shared functionalities across various modules, promoting consistency, reusability, and ease of development.

### 1. `BaseModel` (`chemtools/base/base_models.py`)

The `BaseModel` is the cornerstone for all statistical and machine learning models within `chemtools`. It provides a standardized structure for model initialization, data handling, and result summarization.

**Logic and Purpose:**

*   **Abstract Base Class (ABC):** `BaseModel` is an abstract class, meaning it cannot be instantiated directly. Instead, concrete model implementations (e.g., [`PrincipalComponentAnalysis`](exploration/principal-component-analysis.md), [`LinearRegression`](regression/Linear%20Regression.md), [`OneWayANOVA`](stats/anova.md)) must inherit from `BaseModel` and implement its abstract methods.
*   **Common Functionalities:**
    *   **`save(filename)` / `load(filename)`:** Standardized methods for serializing and deserializing model instances using `pickle`, allowing models to be saved and reloaded for later use without re-fitting.
    *   **`model_name` & `method`:** Attributes to clearly identify the specific model and statistical method being used.
    *   **`notes`:** A list to store important notes, warnings, or contextual information generated during model fitting or analysis.
    *   **Summary Generation (`@property summary`):** A powerful property that automatically generates a well-formatted, human-readable summary string of the model's results. This summary is highly customizable and structured into sections (general info, coefficients, tables, additional statistics, notes).
    *   **`_get_summary_data()` (Abstract Method):** This is the core abstract method that every subclass *must* implement. It is responsible for collecting all relevant model-specific results (e.g., eigenvalues, coefficients, F-statistics) and returning them in a structured dictionary format. `BaseModel` then takes this dictionary and formats it into the final summary string.
    *   **Summary Formatting Helpers:** A suite of private helper methods (`_format_separator`, `_format_title`, `_get_datetime_info`, `_calculate_multicolumn_widths`, `_format_multicolumn_dict`, `_calculate_table_col_widths`, `_format_table`, `_wrap_text`, `_format_notes`) handle the intricate details of text formatting, column alignment, and text wrapping to produce a clean and consistent summary output.

**Illustration of Logic:**

When a user calls `print(model.summary)` on any `chemtools` model, the `BaseModel`'s `summary` property is activated. It calls the model-specific `_get_summary_data()` to retrieve the raw results, then uses its internal formatting logic to present these results in a standardized, easy-to-read text report. This ensures that regardless of the model, the output format is predictable and professional.

### 2. `BasePlotter` (`chemtools/plots/base.py`)

The `BasePlotter` class provides a unified and flexible framework for generating plots across different visualization backends (Matplotlib and Plotly). It abstracts away the complexities of backend-specific styling and figure creation, allowing plot subclasses to focus on the data visualization logic.

**Logic and Purpose:**

*   **Backend Agnostic Interface:** `BasePlotter` allows users to choose their preferred plotting library (`matplotlib` or `plotly`) at initialization, and all subsequent plotting methods in subclasses will adapt accordingly.
*   **Theming and Styling:**
    *   **`THEMES`:** Defines comprehensive color palettes and aesthetic settings for "light" and "dark" themes.
    *   **`STYLE_PRESETS`:** Provides predefined style configurations (e.g., "default", "minimal", "presentation") for both Matplotlib and Plotly, enabling quick application of visual styles.
    *   **`_init_matplotlib_style()` / `_init_plotly_style()`:** Methods to configure the chosen backend's global or template settings based on the selected theme and style preset.
    *   **Automatic Colormap Generation:** Automatically creates sequential, diverging, and raw colormaps from the theme's categorical color palette, making them readily available for continuous data plots.
*   **Common Plotting Utilities:**
    *   **`_process_common_params(**kwargs)`:** A crucial method that processes and normalizes common plotting parameters (e.g., `figsize`, `title`, `xlabel`, `ylabel`, `showlegend`) passed by the user. This ensures consistency across all plot types and handles backend-specific conversions (e.g., Matplotlib `figsize` to Plotly `width`/`height`).
    *   **`_create_figure(**kwargs)`:** Handles the creation of the base figure and axes object(s) for the chosen backend, applying initial theme and style settings.
    *   **`_set_labels(ax_or_fig, ...)`:** A unified method for setting axis labels and titles, adapting to Matplotlib's `ax.set_xlabel` / `fig.suptitle` and Plotly's `fig.update_layout`.
    *   **`_apply_common_layout(fig_or_ax, params)`:** Applies final layout adjustments, including the main figure title, watermark, and `tight_layout` (for Matplotlib), ensuring a polished look.
    *   **`add_watermark(fig, text, alpha)`:** Allows adding a customizable watermark to plots.

**Illustration of Logic:**

When a user creates a plot using any `chemtools` plotting class (e.g., `HistogramPlot`, [`DimensionalityReductionPlot`](dimensional_reduction/famd.md)), they can specify the `library`, `theme`, and `style_preset`. The `BasePlotter` handles all the setup: it configures the chosen backend's aesthetics, processes common plot parameters, creates the figure, sets labels, and applies a consistent final layout. This means plot developers only need to implement the core data visualization logic within their specific plot methods, relying on `BasePlotter` for all the boilerplate and styling.

## Conclusion

By leveraging `BaseModel` and `BasePlotter`, `chemtools` provides a robust, consistent, and user-friendly environment for chemometrics. This architectural design ensures that new models and plotting functionalities can be added seamlessly while maintaining a high standard of code quality and user experience.
