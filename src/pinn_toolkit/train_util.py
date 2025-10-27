"""
Utility functions for training, used to compute validation losses

This module provides a suite of functions for two primary purposes:
1.  **Error Computation**: Efficiently calculates the L2 error between a trained
    PINN model and a reference dataset, with robust handling for large datasets
    through chunking.
2.  **Parameter Generation**: Creates sets of input parameters for model
    evaluation using various sampling strategies (e.g., grid, random,
    hypercube).
"""

# ==============================================================================
# 1. Imports
# ==============================================================================

# Core libraries
import warnings
from functools import partial

# JAX and related libraries
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, random, vmap

# Local modules
from .sampler import Sampler
from .util import L2  # Assuming L2 is in your util module

# ==============================================================================
# 2. Error Computation
# ==============================================================================


@eqx.filter_jit
def compute_err_core(params, static, ref_data, chunk_size):
    """
    Computes L2 error in a memory-efficient, JIT-compiled manner.

    This core function iterates through chunks of a reference dataset using
    `jax.lax.scan` to avoid memory overflow when dealing with a large number
    of simulations. It calculates the L2 error for each output variable.

    Args:
        params: The trainable parameters of the Equinox model.
        static: The static components of the Equinox model.
        ref_data (dict): A dictionary containing the reference data. It must
            include 'x' and 't' grids, simulation parameters, and corresponding
            output solutions.
        chunk_size (int): The number of simulations to process in each chunk.

    Returns:
        dict: A dictionary mapping each output variable to its mean L2 error
              across all simulations (e.g., {'L2_u': 0.01, 'L2_v': 0.02}).
    """
    # 1. Reconstruct the model and identify input/output keys
    model = eqx.combine(params, static)
    inp_idx, out_idx = model.inp_idx, model.out_idx
    param_keys_ordered = sorted([k for k in inp_idx if k not in ['x', 't']], key=inp_idx.get)
    out_keys_ordered = sorted(out_idx.keys(), key=out_idx.get)

    # 2. Set up spatial and temporal grids
    x_ref, t_ref = ref_data['x'], ref_data['t']
    nx, nt = x_ref.shape[0], t_ref.shape[0]
    X_flat, T_flat = map(jnp.ravel, jnp.meshgrid(x_ref, t_ref, indexing="xy"))

    # 3. Prepare for the scan loop over chunks
    num_simulations = ref_data[param_keys_ordered[0]].shape[0]
    n_chunks = num_simulations // chunk_size

    # 4. Define the computation for a single chunk
    def compute_chunk_scan_step(carry, idx):
        """Processes one chunk of data to update the running error sum."""
        start = idx * chunk_size
        # Slice the parameters and reference outputs for the current chunk
        param_chunks = tuple(
            lax.dynamic_slice(ref_data[key], (start,), (chunk_size,))
            for key in param_keys_ordered
        )
        ref_output_chunks = tuple(
            lax.dynamic_slice(ref_data[key], (start, 0, 0), (chunk_size, nx, nt))
            for key in out_keys_ordered
        )

        # Vmap the model evaluation over the chunk of parameters
        def eval_single_param_set(*params_single):
            preds_flat = vmap(model, in_axes=(0, 0, *(None,) * len(param_keys_ordered)))(
                X_flat, T_flat, *params_single
            )
            return jax.tree.map(lambda arr: arr.reshape(nx, nt), preds_flat)

        pred_outputs = vmap(eval_single_param_set)(*param_chunks)

        # Compute L2 error for each simulation in the chunk and sum them
        err_vectors = jax.tree.map(vmap(L2), pred_outputs, ref_output_chunks)
        chunk_sums = jax.tree.map(jnp.sum, err_vectors)
        new_carry = jax.tree.map(lambda r, c: r + c, carry, chunk_sums)
        return new_carry, None

    # 5. Execute the scan and calculate the final mean errors
    initial_carry = tuple([0.0] * len(out_keys_ordered))
    total_sums, _ = lax.scan(compute_chunk_scan_step, initial_carry, jnp.arange(n_chunks))

    mean_errors = {
        f"L2_{key}": total_sums[i] / num_simulations
        for i, key in enumerate(out_keys_ordered)
    }
    return mean_errors


def compute_err(params, static, ref_data, chunk_size=1):
    """
    User-facing wrapper to compute L2 error with automatic data truncation.

    This function ensures that the number of simulations in `ref_data` is
    perfectly divisible by `chunk_size`. If not, it issues a warning and
    truncates the dataset to the largest possible size that allows for even
    chunking before passing it to the JIT-compiled `compute_err_core`.

    Args:
        params: The trainable parameters of the Equinox model.
        static: The static components of the Equinox model.
        ref_data (dict): A dictionary of reference data. Keys that are not
            'x' or 't' and have a first dimension are treated as simulation data.
        chunk_size (int): The desired number of simulations to process per batch.

    Returns:
        dict: A dictionary of mean L2 errors for each output variable.
    """
    # 1. Robustly determine the number of simulations from the reference data
    num_simulations = next(
        (
            val.shape[0] for key, val in ref_data.items()
            if key not in ['x', 't'] and hasattr(val, 'shape') and val.shape
        ),
        None
    )

    if num_simulations is None:
        # If no simulation data is found, run with the original data
        return compute_err_core(params, static, ref_data, chunk_size)

    # 2. Truncate data if the number of simulations is not divisible by chunk_size
    chunk_size = max(1, int(chunk_size))
    remainder = num_simulations % chunk_size

    if remainder == 0:
        return compute_err_core(params, static, ref_data, chunk_size)

    new_num_simulations = num_simulations - remainder
    warnings.warn(
        f"The number of simulations ({num_simulations}) is not divisible by "
        f"chunk_size ({chunk_size}). Truncating data to {new_num_simulations} "
        f"simulations for this run.",
        UserWarning
    )

    # 3. Create a truncated copy of the reference data
    ref_data_truncated = {
        key: (
            leaf[:new_num_simulations]
            if hasattr(leaf, 'shape') and leaf.shape and leaf.shape[0] == num_simulations
            else leaf
        )
        for key, leaf in ref_data.items()
    }

    return compute_err_core(params, static, ref_data_truncated, chunk_size)


# ==============================================================================
# 3. Parameter Generation
# ==============================================================================


def grid(size, span_model):
    """
    Generates a full factorial grid of points from a dictionary of spans.

    Args:
        size (int): The number of points to generate along each dimension.
        span_model (dict): A dictionary mapping parameter names to their
                           `(min, max)` range.

    Returns:
        dict: A dictionary mapping parameter names to flattened arrays of
              grid points.
    """
    vals_list = [jnp.linspace(min_val, max_val, size) for min_val, max_val in span_model.values()]
    grid_vals = jnp.meshgrid(*vals_list, indexing='ij')
    return {key: val.ravel() for key, val in zip(span_model.keys(), grid_vals)}


def grid_inner(size, span_model):
    """
    Generates a grid of points centered within bins.

    This is useful for ensuring samples are not taken at the boundaries.

    Args:
        size (int): The number of bins (and points) along each dimension.
        span_model (dict): A dictionary mapping parameter names to their
                           `(min, max)` range.

    Returns:
        dict: A dictionary mapping parameter names to flattened arrays of
              grid points.
    """
    vals_list = []
    for min_val, max_val in span_model.values():
        step = (max_val - min_val) / size
        start = min_val + step / 2.0
        end = max_val - step / 2.0
        vals_list.append(jnp.linspace(start, end, size))

    grid_vals = jnp.meshgrid(*vals_list, indexing='ij')
    return {key: val.ravel() for key, val in zip(span_model.keys(), grid_vals)}


def uniform_random(key, size, span_model):
    """
    Generates uniformly random samples within the specified parameter spans.

    Note: The total number of samples generated is `size` to the power of the
    number of dimensions (len(span_model)).

    Args:
        key: A JAX PRNG key.
        size (int): The base number used to calculate the total number of samples.
        span_model (dict): A dictionary mapping parameter names to their
                           `(min, max)` range.

    Returns:
        dict: A dictionary mapping parameter names to arrays of random samples.
    """
    dims = list(span_model.items())
    num_dims = len(dims)
    n_samples = size ** num_dims
    keys = random.split(key, num_dims)
    
    out = {}
    for (name, (min_val, max_val)), k in zip(dims, keys):
        out[name] = random.uniform(k, (n_samples,), minval=min_val, maxval=max_val)
    return out


def generate_param(key, method, size, span_model):
    """
    Factory function to generate model parameters using a specified method.

    This function acts as a dispatcher, calling the appropriate sampling
    function based on the `method` string. It automatically filters out
    non-parameter keys like 'x' and 't' from the span model.

    Args:
        key: A JAX PRNG key for methods requiring randomness.
        method (str): The sampling method. Supported built-in values are
                      "grid", "grid_inner". Other values (e.g., "latin",
                      "sobol") are passed to `Sampler.sample_hypercube`.
        size (int): The number of points per dimension or total samples,
                    depending on the method.
        span_model (dict): A dictionary mapping all variable names (including
                           'x' and 't') to their `(min, max)` ranges.

    Returns:
        dict: A dictionary of generated parameter sets.
    """
    # Filter out spatial/temporal dimensions to get only model parameters
    filtered_span_model = {k: v for k, v in span_model.items() if k not in ['x', 't']}
    num_params = len(filtered_span_model)

    if method == "grid":
        return grid(size, filtered_span_model)
    elif method == "gridInner":
        return grid_inner(size, filtered_span_model)
    elif method == "uniformRandom":
        return uniform_random(key, size, filtered_span_model)
    else:
        # For other methods, delegate to the more general Sampler class
        n_samples = size ** num_params
        return Sampler.sample_hypercube(
            key, ranges=filtered_span_model, n_samples=n_samples, method=method
        )