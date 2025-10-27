"""
JAX & Equinox Utility Module.

This module provides a collection of helper functions designed to streamline
working with JAX PyTrees and Equinox models. The utilities are organized into
the following sections:

1.  **Core Data & PyTree Manipulation**: Functions for transforming, slicing,
    and analyzing PyTrees (e.g., mapping value ranges, calculating lengths,
    splitting).
2.  **Serialization & Model I/O**: Helpers for saving and loading Equinox models
    and generic PyTrees to/from disk.
3.  **Data Generation & Sampling**: Tools for creating data subsets, such as
    stratified sampling.
4.  **Miscellaneous Utilities**: Other useful helpers, including serialization
    for JAX random keys.
"""

# Standard library
import h5py
import pickle
import warnings
from typing import Any, Dict, List, Tuple, TypeVar

# Third-party: numerical and plotting
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import lax, random, tree_util

# ==============================================================================
# 1. Core Data & PyTree Manipulation
# ==============================================================================

def map_span(u, src, tgt):
    """
    Linearly maps an array of values from a source range to a target range.

    This function performs an affine transformation on the input array `u`
    such that the values in the source range `src` are mapped to the
    corresponding values in the target range `tgt`.

    Args:
        u: The input JAX array to be mapped.
        src: A tuple `(min, max)` representing the source range.
        tgt: A tuple `(min, max)` representing the target range.

    Returns:
        A new JAX array with values mapped to the target range.

    Example:
        >>> key = random.PRNGKey(0)
        >>> data = random.uniform(key, (5,))  # Values between 0 and 1
        >>> src_range = (0.0, 1.0)
        >>> tgt_range = (10.0, 20.0)
        >>> mapped_data = map_span(data, src_range, tgt_range)
        >>> print(mapped_data)
        [14.447989 17.155949 15.289549 15.680962 10.779892]
    """
    a, b = src
    c, d = tgt
    if a == b:
        return jnp.full_like(u, (c + d) / 2.0)
    return (u - a) * (d - c) / (b - a) + c


def map_span_dict(d_src, span_src, span_tgt):
    """
    Maps dictionary values from a source range to a target range.

    This function iterates through a dictionary and applies `map_span` to each
    value where a corresponding key exists in both the source and target span
    dictionaries.

    Args:
        d_src: The source dictionary with JAX array values.
        span_src: A dictionary mapping keys to their source `(min, max)` range.
        span_tgt: A dictionary mapping keys to their target `(min, max)` range.

    Returns:
        A new dictionary with the mapped values.
    """
    return {
        key: map_span(value, span_src[key], span_tgt[key])
        if key in span_src and key in span_tgt
        else value
        for key, value in d_src.items()
    }


def get_i(pytree, idx):
    """
    Extracts the i-th slice from every leaf in a PyTree.

    This is useful for batching or accessing a single data point from a
    PyTree where each leaf is an array representing a collection of data.

    Args:
        pytree: The PyTree to slice. Each leaf must be indexable.
        idx: The index of the slice to extract.

    Returns:
        A new PyTree with the same structure, where each leaf is the i-th
        slice of the corresponding leaf in the input PyTree.

    Example:
        >>> data = {'x': jnp.arange(10), 'y': jnp.ones((10, 2))}
        >>> point_5 = get_i(data, 5)
        >>> print(point_5)
        {'x': Array(5, dtype=int32), 'y': Array([1., 1.], dtype=float32)}
    """
    return tree_util.tree_map(lambda leaf: leaf[idx], pytree)


def get_len(pytree):
    """
    Returns the length of PyTree leaves, checking for consistency.

    The length is defined as the size of the first dimension of each leaf array.
    This function raises an error if the leaves have inconsistent lengths.

    Args:
        pytree: The PyTree to measure.

    Returns:
        The common length of the leaves.

    Raises:
        ValueError: If the leaves of the PyTree have inconsistent first-dimension
            lengths.
        TypeError: If a leaf is not array-like or does not have a shape.
    """
    leaves = tree_util.tree_leaves(pytree)
    if not leaves:
        return 0

    try:
        lengths = {leaf.shape[0] for leaf in leaves}
    except (AttributeError, IndexError) as e:
        raise TypeError(
            "All leaves of the pytree must be array-like and have a non-empty shape."
        ) from e

    if len(lengths) > 1:
        raise ValueError(f"Pytree leaves have inconsistent lengths: {sorted(list(lengths))}")

    return lengths.pop()


def get_range(pytree):
    """
    Computes the (min, max) range for each leaf in a PyTree.

    Args:
        pytree: The input PyTree with numerical array leaves.

    Returns:
        A new PyTree with the same structure, where each leaf is a tuple
        containing the (min, max) values of the corresponding input leaf.
    """
    min_max_func = lambda arr: (float(jnp.min(arr)), float(jnp.max(arr)))
    return tree_util.tree_map(min_max_func, pytree)


def L2(pytree1, pytree2):
    """
    Computes the Root Mean Square Error (RMSE) between two PyTrees.

    The RMSE is calculated over all elements in the PyTrees. Both PyTrees
    must have the same structure and leaf shapes.

    Args:
        pytree1: The first PyTree (e.g., predictions).
        pytree2: The second PyTree (e.g., ground truth).

    Returns:
        A scalar JAX array representing the RMSE.
    """
    sq_diffs = tree_util.tree_map(lambda a, b: jnp.sum((a - b) ** 2), pytree1, pytree2)
    total_sq_error = jnp.sum(jnp.array(tree_util.tree_leaves(sq_diffs)))
    num_elements = sum(a.size for a in tree_util.tree_leaves(pytree1))
    return jnp.sqrt(total_sq_error / num_elements)


def split_pytree(pytree, n_splits):
    """
    Splits each leaf of a PyTree into `n_splits` parts along the first axis.

    Args:
        pytree: The PyTree to split.
        n_splits: The number of PyTrees to create.

    Returns:
        A list of PyTrees, where each is a chunk of the original.
    """
    leaves, treedef = tree_util.tree_flatten(pytree)
    split_leaves = [jnp.array_split(leaf, n_splits) for leaf in leaves]

    # Transpose the list of lists to group leaves for each new PyTree
    # e.g., [[a1, a2], [b1, b2]] -> [(a1, b1), (a2, b2)]
    rearranged_leaves = zip(*split_leaves)

    return [tree_util.tree_unflatten(treedef, leaf_group) for leaf_group in rearranged_leaves]


def tree_to_f32(tree):
    """
    Converts all floating-point leaves in a PyTree to float32.

    Args:
        tree: The PyTree to convert.

    Returns:
        A new PyTree with all floating-point arrays converted to float32.
    """
    def to_f32(x):
        is_float = hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        return x.astype(jnp.float32) if is_float else x
    return tree_util.tree_map(to_f32, tree)


# ==============================================================================
# 2. Serialization & Model I/O
# ==============================================================================


def save_model(model, filename, msg = False):
    """
    Saves an Equinox model to a binary file.

    This is a wrapper around `equinox.tree_serialise_leaves`.

    Args:
        model: The Equinox model to save.
        filename: The path to the file where the model will be saved.
        msg: If True, prints a confirmation message.
    """
    eqx.tree_serialise_leaves(filename, model)
    if msg:
        print(f"Model saved successfully to '{filename}'")


def load_model(skeleton_model, filename, msg = False):
    """
    Loads an Equinox model from a binary file into a skeleton model.

    This is a wrapper around `equinox.tree_deserialise_leaves`.

    Args:
        skeleton_model: A model instance with the same structure as the one
            being loaded. Its weights will be overwritten.
        filename: The path to the file from which to load the model.
        msg: If True, prints a confirmation message.

    Returns:
        The model with loaded weights.
    """
    loaded_model = eqx.tree_deserialise_leaves(filename, skeleton_model)
    if msg:
        print(f"Model loaded successfully from '{filename}'")
    return loaded_model


def save_pytree(pytree, filename):
    """
    Saves a generic PyTree to a binary file using pickle.

    Args:
        pytree: The PyTree to save.
        filename: The path to the output file.
    """
    with open(filename, "wb") as f:
        leaves, treedef = tree_util.tree_flatten(pytree)
        pickle.dump((leaves, treedef), f)


def load_pytree(filename):
    """
    Loads a generic PyTree from a binary file saved using `save_pytree`.

    Ensures that all loaded arrays are converted to JAX arrays.

    Args:
        filename: The path to the file to load.

    Returns:
        The reconstructed PyTree.
    """
    with open(filename, "rb") as f:
        leaves, treedef = pickle.load(f)
    # Ensure all loaded arrays are JAX arrays for consistency
    leaves = [jnp.array(leaf) for leaf in leaves]
    return tree_util.tree_unflatten(treedef, leaves)

def load_h5(filename, indices = None, data_only = True):
    """
    Loads specified datasets from an HDF5 file into a dictionary of JAX arrays.

    This function can load all entries, a single entry by index, or a list of
    indices. It automatically handles different dataset dimensions and converts
    the final data to JAX arrays with gradients stopped.

    Args:
        filename (str): The path to the HDF5 file.
        indices (int or list, optional): The index or indices to load. If None,
                                          all entries are loaded. Defaults to None.
        data_only (bool): If True, utility datasets like 'solve_times' are skipped.
                          Defaults to True.

    Returns:
        A dictionary mapping dataset names to their corresponding JAX array data.
    """
    data_dict = {}
    with h5py.File(filename, 'r') as hf:
        indices_to_load = slice(None) if indices is None else indices

        for key in hf.keys():
            if data_only and key == 'solve_times':
                continue

            output_key = key.removesuffix('_pde')
            dataset = hf[key]

            if dataset.ndim > 1 or (dataset.ndim == 1 and key not in ['x_nd', 't_nd']):
                data = dataset[indices_to_load, ...]
            else:
                data = dataset[:]
            data_dict[output_key] = data

    for key, value in data_dict.items():
        data_dict[key] = jax.lax.stop_gradient(jnp.asarray(value))

    if 'x_nd' in data_dict: data_dict['x'] = data_dict.pop('x_nd')
    if 't_nd' in data_dict: data_dict['t'] = data_dict.pop('t_nd')

    return data_dict
    
# ==============================================================================
# 3. Data Generation & Sampling
# ==============================================================================


def stratified_subset(key, pytree, total_size, num_strata, n):
    """
    Selects a stratified random subset from a PyTree.
    
    **How it works:**
    This function assumes the input `pytree` is already ordered in a meaningful
    way. It operates by dividing the index range `[0, total_size - 1]` into
    `num_strata` contiguous, equal-sized blocks. For this to be effective,
    the data in the `pytree` should be arranged such that these blocks
    correspond to meaningful partitions of the data (e.g., different phases
    of a simulation, different experimental batches, or sequential time periods).

    From each of these index blocks (strata), the function randomly samples `n`
    indices without replacement. The final output is a new PyTree constructed
    from these collected indices, guaranteeing a balanced sample from all
    predefined sections of the original dataset.
    
    Args:
        key: A JAX PRNG key for reproducibility.
        pytree: The PyTree containing the dataset.
        total_size: The total number of items in the dataset.
        num_strata: The number of strata to divide the dataset into.
        n: The number of samples to draw from each stratum.

    Returns:
        A new PyTree containing the stratified random subset.

    Raises:
        ValueError: If `total_size` is not divisible by `num_strata`.
    """
    if total_size % num_strata != 0:
        raise ValueError(
            f"total_size ({total_size}) must be divisible by num_strata ({num_strata})."
        )

    stratum_size = total_size // num_strata
    samples_per_stratum = n

    if n > stratum_size:
        warnings.warn(
            f"Requested n ({n}) samples per stratum, but each stratum only has "
            f"size {stratum_size}. Sampling all {stratum_size} items instead.",
            UserWarning
        )
        samples_per_stratum = stratum_size

    @jax.jit
    def get_indices(k: jax.random.PRNGKey) -> jnp.ndarray:
        """JIT-compiled function to generate stratified indices."""
        stratum_starts = jnp.arange(num_strata) * stratum_size

        def sample_stratum(i: int) -> jnp.ndarray:
            """Samples `n` indices from a single stratum."""
            stratum_key = random.fold_in(k, i)
            start_index = stratum_starts[i]
            perm = random.permutation(stratum_key, stratum_size)
            # Take the first `n` elements from the permutation
            selected_local_indices = lax.dynamic_slice(perm, (0,), (samples_per_stratum,))
            return selected_local_indices + start_index

        all_indices = jax.vmap(sample_stratum)(jnp.arange(num_strata))
        return all_indices.flatten()

    # Generate and apply the indices to the pytree
    indices = get_indices(key)
    return tree_util.tree_map(lambda leaf: leaf[indices], pytree)


# ==============================================================================
# 4. Miscellaneous Utilities
# ==============================================================================


def key_to_hex(key: jax.random.PRNGKey) -> str:
    """
    Serializes a JAX PRNG key to a hexadecimal string.

    This is useful for saving a PRNG state in a human-readable format.

    Args:
        key: The JAX PRNG key to serialize.

    Returns:
        A hexadecimal string representation of the key.
    """
    return ''.join(f'{x:08x}' for x in key)


def hex_to_key(s: str) -> jax.random.PRNGKey:
    """
    Deserializes a hex string into a JAX PRNG key.

    Args:
        s: The hexadecimal string to convert back into a key.

    Returns:
        The deserialized JAX PRNG key.

    Raises:
        ValueError: If the input string has an invalid length.
    """
    if len(s) % 8 != 0:
        raise ValueError("Invalid seed string length; must be a multiple of 8.")

    chunks = [s[i:i+8] for i in range(0, len(s), 8)]
    arr = jnp.array([int(chunk, 16) for chunk in chunks], dtype=jnp.uint32)
    return arr