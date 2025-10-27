# Example: Allen-Cahn and Cahn-Hilliard (ACCH) System

This directory contains a complete example of using the `pinn_toolkit` to train the parameterized parameterized PINN for the 1D coupled Allen-Cahn and Cahn-Hilliard (ACCH) equations. Note that the user are expected to define their own `model.py`, 'residual.py`.

---

## File Breakdown

The key files in this example are:
-   **`model.py`**: Defines the core `PINN` class, which is the neural network itself. It is built as an `equinox.Module`, integrating seamlessly with JAX's functional paradigm. **This is the main module to customize for new mode architectures.**
-   **`residual.py`**: Defines the physics of the problem by translating PDEs, ICs, and BCs into trainable residuals. **This is the main module to customize for new PDEs.**
-   **`pde_dimless.py`**: A PDE class defining the non-dimensionalized ACCH system, built on `diffrax`. Its primary role is to generate ground-truth data for training and validation.
-   **`interactive_pde_suite.py`**: A powerful tool for real-time exploration of a Cahn-Hilliard/Allen-Cahn PDE system. Can be used to compare the numerical PDE solution against a trained PINN model to visually assess the model's accuracy across different physical parameters.
-   **`generate_data.py`**: A module dedicated to solving the 1D ACCH PDE system under different parameters (e.g. `L`: mobility parameter, `M`: diffusivity parameter) using parallel computing.
-   **`train.ipynb`**: The main training notebook. It shows how to import components from `pinn_toolkit` to define the model, set up the loss functions, and run the training engine.
-   **`validation.ipynb`**: An example notebook that uses the analysis suite from `pinn_toolkit/validation.py` to evaluate trained models and generate plots comparing multiple runs.
-   **Other Notebooks**: Includes notebooks for generating data (`generate_data.ipynb`) and interactively exploring the PDE solutions (`interactive_pde_demo.ipynb`).
-   

`_archive` contains code that are outdated or incomplete. It'll be fixed in the future.
---

## How to Run This Example

It is assumed you have already followed the installation instructions in the main `README.md` file and have activated the `jax-pinn-env` Conda environment.

1.  ** (Required) Generate Your Own Data:** Please run the cells in `generate_data.ipynb` to generate the data for validation. Otherwise disable the validation option to avoid any potenital errors. The robustness will be improved in the future.

2.  **Train the Model:** Open and run the cells in `train.ipynb` from top to bottom. This will train a new PINN model and save the results (model weights and loss history) into the `pdes/ACCH/models/` directory.

3.  **Analyze the Results:** Once training is complete, run the cells in `validation.ipynb` to see a detailed performance analysis of the model you just trained, some sample plots are stored inside plots.
