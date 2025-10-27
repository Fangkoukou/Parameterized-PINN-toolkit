# Parameter-Dependent PINN Toolkit in JAX

A lightweight JAX/Equinox framework for researchers to train and evaluate parameterized Physics-Informed Neural Networks (PINNs). Its low-level design offers maximum flexibility for novel research, providing a suite of tools for model definition, training, and large-scale analysis.

This framework is in an early development stage. Its components are intentionally low-level to prioritize maximum flexibility for novel research. As this project originated as a toolkit for our own PINN studyings, some functionalities are currently tailored to those specific problems. We aim to generalize and make the framework more robust in the future.

---

## Key Features

- **Flexible, Low-Level Design**: The framework does not hide details behind high-level abstractions, giving researchers full control over model architecture, loss functions, and training loops.
- **High-Performance JAX Backend**: The entire training engine is built on `jax.lax.scan` and `jax.vmap`, and is JIT-compiled for maximum performance on modern accelerators (GPUs/TPUs).

---

## Getting Started

### Prerequisites
- Anaconda or Miniconda
- Python 3.11+
- An NVIDIA GPU with CUDA drivers installed (for GPU support)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Fangkoukou/Parameterized-PINN-toolkit.git
    cd Parameterized-PINN-toolkit
    ```

2.  **Create and activate the Conda environment:**
    This command creates a new environment named `jax-pinn-env` with all necessary dependencies from the `environment.yml` file, including GPU-enabled JAX.
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

## Project Structure and Usage

This repository is organized into a reusable library (`src/pinn_toolkit`) and an example application that uses it (`pdes/ACCH`).

### 1. The Core Library: `src/pinn_toolkit`

This is the core, reusable framework. It provides a set of modular tools for building and training PINNs.

#### `util.py`: Core Utilities
A collection of helper functions for data manipulation, I/O, and reproducibility.
- **PyTree Utilities**: Functions for mapping value ranges, pytree accessing, slicing, batching, stratified sampling etc.
- **I/O**: Helpers for saving/loading Equinox models, generic PyTrees, and HDF5 datasets.
- **Reproducibility**: JAX PRNG key serialization (into hex string).

#### `sampler.py`: Point Cloud Generation
Provides a collection of methods for generating point clouds for training.
- **1D Sampling**: Generates points on an interval with a uniform grid, random shift (`get`), or per-point noise (`get_jitter`).
- **Hypercube Sampling**: Core utility for sampling in D-dimensional parameter spaces using `uniform`, `sobol`, `halton`, or `latin_hypercube` methods.

#### `derivative.py`: Dynamic Derivative Factory
A factory for dynamically creating scaled and vectorized derivative functions.
- **String-Based API**: Define arbitrary partial derivatives with an intuitive string (e.g., `'phi_x2_t'`).
- **Automated Differentiation**: Composes `jax.grad` to automatically compute any order derivative.
- **Coordinate Scaling**: Automatically applies the chain rule to convert derivatives from normalized to physical coordinates.
- **Automatic Vectorization**: All generated functions are wrapped in `jax.vmap` to handle scalar and batched inputs seamlessly.

#### `train.py`: General-Purpose Training Engine
A high-performance, stateless training engine built on `jax.lax.scan`.
- **JIT-Compiled Loop**: The entire training process is compiled into a single XLA graph for maximum performance.
- **Callback-Driven**: The generic loop is controlled by user-defined functions for data sampling, loss weighting, and validation.
- **Scheduled Operations**: Efficiently run expensive operations (resampling, logging) periodically instead of every step.
- **Robust Model Selection**: Tracks the best model based on the geometric mean of loss components.
- **Asynchronous Logging**: Uses `io_callback` for rich logging without slowing down the GPU/TPU computation.

#### `train_tuil.py`: Training utilities
A temporary module to perform parameter generation and model validation during training. This module will be integrated into other modules in the future.
- **Parameter Set Generation**: Contains a factory `generate_param` for creating structured sets of input parameters.
- **Memory-Efficient Error Computation**: Includes `compute_err`, to calculate L2 error against large reference datasets.

#### `validation.py`: Analysis & Visualization Suite
A comprehensive suite for post-hoc analysis of large-scale experiments.
- **Model Discovery**: Automatically finds and filters trained models from a results directory.
- **Model Evaluation**: A memory-efficient, JIT-compiled evaluator to compute L1, L2, and Linf errors against reference data.
- **Comparative Visualization**: Generates publication-quality plots including error dot plots, individual loss histories, and aggregated loss curves.

### 2. The Example Application: `pdes/ACCH`

This folder demonstrates how to **use** the `pinn_toolkit` to solve a specific problem: the coupled 1D Allen-Cahn and Cahn-Hilliard (ACCH) equations. Given most if not all innovations/studies on PINN focus on model architecture, residual and loss functions, and training strategy, we believe it is best for the users to specify their own ``model`` and ``residual`` modules. A generic blueprint is set up for the 1D ACCH equation and can be easily modified to accomodate different PDEs. Please see the files in pdes/ACCH for detail.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
