# Parameterized PINN Toolkit in JAX

A lightweight JAX/Equinox framework for researchers to train and evaluate parameterized Physics-Informed Neural Networks (PINNs). Its low-level design offers maximum flexibility for novel research, providing a suite of tools for model definition, training, and large-scale analysis.

This framework is in an early development stage. As it originated as a toolkit for our own PINN investigations, some functionalities are currently tailored to those specific problems. We aim to generalize and make the framework more robust in the future.

---

## Getting Started

### Installation

This project uses Conda for environment management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Fangkoukou/Parameterized-PINN-toolkit.git
    cd Parameterized-PINN-toolkit
    ```

2.  **Create and activate the Conda environment:**
    This command creates a new environment named `jax-pinn-env` with all necessary dependencies from the `environment.yml` file.
    ```bash
    conda env create -f environment.yml
    conda activate jax-pinn-env
    ```

3.  **Install the toolkit in editable mode:**
    This command makes your `pinn_toolkit` library importable within the environment and ensures any changes you make to the source code are immediately reflected.
    ```bash
    pip install -e .
    ```

---

## Project Structure

This repository is organized into two main parts: a reusable library (`src/pinn_toolkit`) and an example application (`pdes/ACCH`).

### 1. The Library: `src/pinn_toolkit`

This is the core, reusable framework. It provides a set of modular tools for building and training PINNs.

-   **`model.py`**: Defines the `PINN` class, a configurable MLP built with Equinox.
-   **`sampler.py`**: Provides tools for generating point clouds for training using grid, random, and quasi-random methods.
-   **`derivative.py`**: A powerful factory for dynamically creating scaled and vectorized derivative functions from simple strings (e.g., `'phi_x2_t'`).
-   **`residual.py`**: Defines the physics of the problem by translating PDEs, ICs, and BCs into trainable loss components. **This is the main module to customize for new PDEs.**
-   **`train.py`**: A high-performance, JIT-compiled training engine built on `jax.lax.scan` that is controlled by user-provided callback functions.
-   **`validation.py`**: A comprehensive suite for post-hoc analysis, including batch evaluation and generation of publication-quality comparison plots.
-   **`util.py`**: A collection of core helper functions for data manipulation, serialization, and reproducibility.

### 2. The Example: `pdes/ACCH`

This folder demonstrates how to **use** the `pinn_toolkit` to solve a specific problem: the coupled Allen-Cahn and Cahn-Hilliard (ACCH) equations.

-   **`pde_dimless.py`**: A high-fidelity numerical solver for the ACCH system, used to generate ground-truth data for training and validation.
-   **`train.ipynb`**: The main training notebook. It shows how to import components from `pinn_toolkit` to define the model, set up the loss functions, and run the training engine.
-   **`validation.ipynb`**: An example notebook that uses the analysis suite from `pinn_toolkit/validation.py` to evaluate trained models and generate plots.
-   **`_archive/`, `data/`, `models/`, `plots/`**: Folders containing experimental code, generated data, saved models, and plots for this specific PDE problem. These are ignored by Git.

---

## A Typical Workflow

The framework is designed to connect the library and the example in a clear workflow:

1.  **Define the Problem:** In `pdes/ACCH/`, the `pde_dimless.py` script defines the specific physics of the Allen-Cahn/Cahn-Hilliard system.

2.  **Configure and Train:** The `pdes/ACCH/train.ipynb` notebook demonstrates the main workflow:
    -   It imports `PINN`, `Residual`, `Derivative`, and `Train` from the `pinn_toolkit`.
    -   It uses these tools to configure a neural network tailored to the ACCH problem.
    -   It defines the loss functions and callbacks required by the training engine.
    -   It launches the training process using `Train.train(...)`.

3.  **Analyze Results:** After training, the `pdes/ACCH/validation.ipynb` notebook shows how to use the functions in `pinn_toolkit/validation.py` to load the saved models, compute error metrics against a reference dataset, and create plots comparing the performance of different training runs.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
