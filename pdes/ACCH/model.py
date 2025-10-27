# -*- coding: utf-8 -*-
"""
Equinox-based Physics-Informed Neural Network (PINN) Module.

This module defines the core PINN model architecture using the Equinox library.
It features a flexible MLP backbone and provides multiple interfaces for evaluation,
including raw positional inputs and user-friendly named dictionary inputs.
The model also supports configurable floating-point precision (FP32/FP64).
"""

# ==============================================================================
# 1. Imports
# ==============================================================================

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import tree_util

# ==============================================================================
# 2. Helper Functions
# ==============================================================================


def tree_to_f32(tree):
    """
    Converts all floating-point array leaves in a PyTree to float32.

    This is a utility function to enforce single precision on a model, which can
    improve performance and reduce memory usage, especially on GPUs.

    Args:
        tree: A JAX PyTree (e.g., an Equinox model).

    Returns:
        A new PyTree with all floating-point arrays converted to `jnp.float32`.
    """
    def to_f32(x):
        is_float_array = hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        return x.astype(jnp.float32) if is_float_array else x
    return tree_util.tree_map(to_f32, tree)

# ==============================================================================
# 3. PINN Model Class
# ==============================================================================


class PINN(eqx.Module):
    """
    A Physics-Informed Neural Network (PINN) model implemented in Equinox.

    This class encapsulates a Multi-Layer Perceptron (MLP) and includes metadata
    for mapping named inputs/outputs and for coordinate transformations. It provides
    several methods for model evaluation, catering to different use cases from
    raw, batched evaluation to named, user-friendly prediction.

    Attributes:
        net (eqx.nn.MLP): The core neural network.
        inp_idx (dict): A mapping from input names to their integer index.
        out_idx (dict): A mapping from output names to their integer index.
        span_pde (dict): A dictionary of (min, max) spans for physical variables.
        span_model (dict): A dictionary of (min, max) spans for model variables.
    """
    net: eqx.nn.MLP
    inp_idx: dict
    out_idx: dict
    span_pde: dict
    span_model: dict

    def __init__(self, inp_idx, out_idx, span_pde, span_model, width, depth, key=None, fp64=False):
        """
        Initializes the PINN model.

        Args:
            inp_idx (dict): Maps input names (e.g., 'x', 't') to their index.
            out_idx (dict): Maps output names (e.g., 'phi', 'c') to their index.
            span_pde (dict): Dictionary of physical parameter spans.
            span_model (dict): Dictionary of model's normalized parameter spans.
            width (int): The width of the hidden layers in the MLP.
            depth (int): The number of hidden layers in the MLP.
            key (jax.random.PRNGKey, optional): A JAX PRNG key for weight initialization.
            fp64 (bool): If `True`, model parameters are float64. If `False` (default),
                         they are converted to float32 for performance.
        """
        key = jax.random.PRNGKey(0) if key is None else key
        self.inp_idx = inp_idx
        self.out_idx = out_idx
        self.span_pde = span_pde
        self.span_model = span_model

        # Define the MLP architecture. Note: Equinox respects JAX's global
        # precision settings, so this will be FP64 if `enable_x64` is True.
        net = eqx.nn.MLP(
            in_size=len(inp_idx),
            out_size=len(out_idx),
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

        # Explicitly convert the network's parameters to float32 if requested.
        if not fp64:
            net = tree_to_f32(net)

        self.net = net

    # ==========================================================================
    # 4. Core Evaluation Methods
    # ==========================================================================

    def __call__(self, *args):
        """
        Performs a raw forward pass for a single set of ordered inputs.

        This is a low-level method primarily for internal use or when performance
        is critical and inputs are already correctly ordered.

        Args:
            *args: Positional arguments corresponding to the ordered inputs of the network.
                   Only the first `len(self.inp_idx)` arguments are used.

        Returns:
            A tuple of the network's outputs.
        """
        # Stack positional arguments into a single input vector for the MLP
        x = jnp.stack(args[:len(self.inp_idx)], axis=-1)
        return tuple(self.net(x))

    def eval(self, *args):
        """
        A flexible forward pass that handles both scalar and batched array inputs.

        This method uses `jax.vmap` to automatically broadcast the computation
        over array inputs while keeping scalar inputs fixed. This is highly
        efficient for evaluating on grids or with mixed parameter types.

        Args:
            *args: A mix of scalar or array-like positional inputs.

        Returns:
            A tuple of output arrays, with shapes corresponding to the broadcasted
            input shapes.
        """
        n = len(self.inp_idx)
        arrs = tuple(jnp.asarray(a) for a in args[:n])

        # If all inputs are scalars, use the simple __call__ method
        if all(jnp.ndim(a) == 0 for a in arrs):
            return self.__call__(*arrs)

        # Otherwise, use vmap for automatic batching/broadcasting
        in_axes = tuple(None if jnp.ndim(a) == 0 else 0 for a in arrs)
        return jax.vmap(self.__call__, in_axes=in_axes)(*arrs)

    # ==========================================================================
    # 5. User-Facing Prediction Methods
    # ==========================================================================

    def predict(self, inp, names=None):
        """
        The primary user-facing method for making predictions with named inputs.

        This method takes a dictionary of named inputs, automatically orders them
        correctly, evaluates the model, and returns a dictionary of named outputs.

        Args:
            inp (dict): A dictionary mapping input names to their values (scalar or array).
            names (list, optional): A list of output names to return. If None, all
                                    outputs are returned.

        Returns:
            A dictionary mapping output names to their predicted values.
        """
        # Order the input values from the dictionary according to `inp_idx`
        sorted_inp_names = sorted(self.inp_idx.keys(), key=self.inp_idx.get)
        ordered_args = [inp[name] for name in sorted_inp_names]

        # Evaluate the model with the ordered inputs
        outs = self.eval(*ordered_args)

        # Package the tuple of outputs into a named dictionary
        keys = names or list(self.out_idx.keys())
        return {k: outs[self.out_idx[k]] for k in keys}

    def validation(self, x, t, *scalars):
        """
        A specialized method to evaluate the network on a full space-time grid.

        This is a convenience function for generating 2D solution fields for
        plotting or comparison with numerical solvers.

        Args:
            x (jnp.ndarray): A 1D array of spatial coordinates.
            t (jnp.ndarray): A 1D array of temporal coordinates.
            *scalars: Additional scalar parameters (e.g., L, M) to be held constant
                      across the grid.

        Returns:
            A dictionary containing the input grids ('x', 't') and the reshaped
            output fields ('phi', 'c').
        """
        # Create a 2D mesh from the 1D coordinate arrays
        X, T = jnp.meshgrid(x, t, indexing="xy")
        flat_inp = [X.ravel(), T.ravel(), *scalars]

        # Evaluate the model on the flattened grid
        outs = self.eval(*flat_inp)

        # Reshape the flat outputs back to the 2D grid shape
        return {
            "x": x,
            "t": t,
            "phi": outs[0].reshape(X.shape),
            "c": outs[1].reshape(X.shape),
        }