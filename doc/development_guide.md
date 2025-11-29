# Development Guide

This guide provides comprehensive instructions for setting up your development environment, installing dependencies, and contributing to the `chemtools` project.

## Table of Contents
1.  [Prerequisites](#prerequisites)
2.  [Setting Up Your Environment](#setting-up-your-environment)
    *   [Using `uv` (Recommended)](#using-uv-recommended)
    *   [Using Conda/Miniforge](#using-condaminiforge)
3.  [Installing Dependencies](#installing-dependencies)
4.  [Running Tests](#running-tests)
5.  [Code Style and Linting](#code-style-and-linting)
6.  [Building Documentation](#building-documentation)
7.  [Contributing](#contributing)

## 1. Prerequisites

Before you begin, ensure you have the following installed:

*   **Git**: For version control. [Download Git](https://git-scm.com/downloads).
*   **Python 3.12**: The project is developed with Python 3.12.
    *   **For `uv` users**: `uv` will automatically find and use an installed Python interpreter.
    *   **For Conda/Miniforge users**: Python will be installed as part of the environment setup.

## 2. Setting Up Your Environment

You have two primary options for setting up your development environment: `uv` (recommended for speed and efficiency) or Conda/Miniforge (if you prefer a more comprehensive environment manager, especially for scientific computing).

### Using `uv` (Recommended)

`uv` is an extremely fast Python package installer and resolver, written in Rust.

1.  **Install `uv`**:
    ```bash
    pip install uv
    # Or, if you have Rust/Cargo installed:
    # cargo install uv
    ```
    For detailed installation instructions, refer to the official [uv documentation](https://docs.astral.sh/uv/installation/).

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/MGriot/chemtools.git
    cd chemtools
    ```

3.  **Create and Activate a Virtual Environment**:
    ```bash
    # IMPORTANT: If you encounter permissions errors (e.g., "Access is denied") during this step,
    # it means a file in the `.venv` directory is locked by another process. You may need to
    # manually delete the `.venv` directory before running this command.
    # On Windows, ensure no terminal is open in `.venv` and Python processes are closed.
    # On Linux/macOS, ensure no processes are using the `.venv` directory.
    uv venv --clear
    ```
    To activate the environment:
    *   **On Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source .venv/bin/activate
        ```
    You should see `(.venv)` or similar in your terminal prompt, indicating the environment is active.

### Using Conda/Miniforge

If you prefer using Conda or Miniforge for environment management:

1.  **Install Conda or Miniforge**:
    If you don't have a Conda distribution, we recommend downloading and installing [Miniforge](https://github.com/conda-forge/miniforge#miniforge-installers). Miniforge provides `conda` and defaults to the `conda-forge` channel, which is excellent for scientific Python packages.

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/MGriot/chemtools.git
    cd chemtools
    ```

3.  **Create and Activate the Environment**:
    The project includes an `environment.yml` file to easily set up the environment with all necessary dependencies.
    ```bash
    conda env create -f environment.yml
    conda activate chemtools
    ```
    You should see `(chemtools)` in your terminal prompt, indicating the environment is active.

## 3. Installing Dependencies

### For `uv` users:

Once your `uv` virtual environment is active, install the project dependencies:
```bash
uv pip install -r requirements.txt
```

### For Conda/Miniforge users:

If you created your environment using `conda env create -f environment.yml`, all dependencies are already installed. If you need to add more packages later, you can do so with:
```bash
conda install -c conda-forge <package-name>
# Or for pip-only packages
pip install <package-name>
```

## 4. Running Tests

It's crucial to run tests to ensure your changes haven't introduced any regressions. The project uses `pytest`.

To run all tests:
```bash
pytest
```

To run a specific test file:
```bash
pytest path/to/your/test_file.py
```

## 5. Code Style and Linting

We use `pylint` for code linting to ensure consistent code style and identify potential issues.

To run pylint:
```bash
pylint chemtools
```

Please ensure your code passes pylint checks before submitting pull requests.

## 6. Building Documentation

The documentation is built using Markdown files. Any changes to the `doc/` directory will be reflected when the documentation is rendered.

## 7. Contributing

We welcome contributions! Please see the `CONTRIBUTING.md` file (if available) for detailed guidelines on how to contribute to the project, report bugs, and suggest features. If `CONTRIBUTING.md` is not present, please open an issue on the GitHub repository to discuss your proposed changes.