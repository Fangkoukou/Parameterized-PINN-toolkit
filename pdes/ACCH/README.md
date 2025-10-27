# Example: Allen-Cahn and Cahn-Hilliard (ACCH) System

This directory contains a complete example of using the `pinn_toolkit` to train the parameterized parameterized PINN for the 1D coupled Allen-Cahn and Cahn-Hilliard (ACCH) equations. Note that the user are expected to define their own `model.py`, 'residual.py`.

---

## File Breakdown

The key files in this example are:

-   **`pde_dimless.py`**: A high-fidelity numerical solver for the ACCH system, built on `diffrax`. Its primary role is to generate the ground-truth data used for training and validation. It uses a dimensionless formulation to ensure numerical stability.

-   **`train.ipynb`**: The main training notebook. This is the best place to start. It demonstrates the complete workflow:
    1.  Importing the necessary modules (`PINN`, `Residual`, `Derivative`, `Train`) from the `pinn_toolkit`.
    2.  Defining the physical parameters and the model architecture.
    3.  Setting up the specific loss functions for the ACCH residuals.
    4.  Configuring and running the main training engine.

-   **`validation.ipynb`**: An example notebook that uses the analysis suite from `pinn_toolkit/validation.py` to evaluate the models trained by `train.ipynb`. It shows how to load results, compute error metrics, and generate comparison plots.

-   **`generate_data.ipynb`**: A utility notebook that uses `pde_dimless.py` to generate the HDF5 reference datasets needed for training and validation.

-   **`interactive_pde_demo.ipynb`**: A notebook with interactive widgets that allows you to explore the ACCH solution space by varying physical parameters and seeing the PINN's predictions in real-time.

-   **`_archive/`, `data/`, `models/`, `plots/`**: Folders containing experimental code, generated data, saved models, and plots for this specific PDE problem. These folders are ignored by Git as per the `.gitignore` file.

---

## How to Run This Example

It is assumed you have already followed the installation instructions in the main `README.md` file and have activated the `jax-pinn-env` Conda environment.

1.  ** (Required) Generate Your Own Data:** Please run the cells in `generate_data.ipynb` to generate the data for validation. Otherwise disable the validation option to avoid any potenital errors. The robustness will be improved in the future.

2.  **Train the Model:** Open and run the cells in `train.ipynb` from top to bottom. This will train a new PINN model and save the results (model weights and loss history) into the `pdes/ACCH/models/` directory.

3.  **Analyze the Results:** Once training is complete, run the cells in `validation.ipynb` to see a detailed performance analysis of the model you just trained, some sample plots are stored inside plots.
