"""
Sampling Utilities for Physics-Informed Neural Networks (PINNs).

This module provides a `Sampler` class containing a collection of static methods
for generating point clouds for training and validation. It supports both
1D grid-based sampling with jitter and multi-dimensional hypercube sampling
using various quasi-random and random sequences.
"""

# ==============================================================================
# 1. Imports
# ==============================================================================

import jax
import jax.numpy as jnp
from jax import random
from scipy.stats import qmc  # Used for quasi-random sequence generation

# ==============================================================================
# 2. Sampler Class
# ==============================================================================


class Sampler:
    """
    A stateless collection of static methods for generating sample points.
    """

    @staticmethod
    def _make_uniform_grid(num, span):
        """
        Creates a deterministic 1D grid of points, including endpoints.

        This is a private helper function.

        Args:
            num (int): The number of points to generate.
            span (tuple): A tuple `(min, max)` defining the interval.

        Returns:
            A JAX array containing the grid points.
        """
        if num == 0:
            return jnp.array([])
        if num == 1:
            # Return the start of the span for a single point
            return jnp.array([span[0]])
        return jnp.linspace(span[0], span[1], num, endpoint=True)

    @staticmethod
    def get_jitter(key, nums, spans, noise=1.0):
        """
        Generates 1D points on uniform grids with per-point jitter.

        This method first creates uniform grids for each specified span and then
        adds a unique, independent random jitter to *every point*. This is useful
        for breaking the regularity of a grid during training.

        Args:
            key: A JAX PRNG key.
            nums (list of int): A list of the number of points for each grid.
            spans (list of tuple): A list of `(min, max)` spans for each grid.
            noise (float): A multiplier for the jitter magnitude. 1.0 corresponds
                           to jitter within one grid cell.

        Returns:
            A single concatenated JAX array of the generated points.

        Example:
            >>> key = random.PRNGKey(0)
            >>> nums = [3, 2]
            >>> spans = [(0, 1), (10, 11)]
            >>> points = Sampler.get_jitter(key, nums, spans)
            >>> print(points)
            [0.22239947 0.8577974  1.2644775  10.038994   11.        ]
        """
        keys = random.split(key, len(nums))
        samples = []
        for k, num, span in zip(keys, nums, spans):
            base = Sampler._make_uniform_grid(num, span)
            dx = (span[1] - span[0]) / (num - 1) if num > 1 else 0.0
            # Generate a unique random value for each point on the base grid
            jitter = random.uniform(k, base.shape, minval=0.0, maxval=dx)
            arr = jnp.clip(base + jitter * noise, span[0], span[1])
            samples.append(arr)
        return jnp.concatenate(samples, axis=0)

    @staticmethod
    def get(key, nums, spans, noise=1.0):
        """
        Generates 1D points on uniform grids with a shared global shift.

        This method creates uniform grids and applies the *same* random shift
        to all points within a given grid. This preserves the relative spacing
        of the points while shifting the entire grid randomly.

        Args:
            key: A JAX PRNG key.
            nums (list of int): A list of the number of points for each grid.
            spans (list of tuple): A list of `(min, max)` spans for each grid.
            noise (float): A multiplier for the shift magnitude. 1.0 corresponds
                           to a shift within one grid cell.

        Returns:
            A single concatenated JAX array of the generated points.

        Example:
            >>> key = random.PRNGKey(0)
            >>> nums = [3, 2]
            >>> spans = [(0, 1), (10, 11)]
            >>> points = Sampler.get(key, nums, spans)
            >>> print(points)
            [0.22239947 0.72239947 1.2223995  10.857797   11.857797 ]
        """
        keys = random.split(key, len(nums))
        samples = []
        for k, num, span in zip(keys, nums, spans):
            base = Sampler._make_uniform_grid(num, span)
            dx = (span[1] - span[0]) / (num - 1) if num > 1 else 0.0
            # Generate a single random shift for the entire grid
            shift = random.uniform(k, (), minval=0.0, maxval=dx)
            arr = jnp.clip(base + shift * noise, span[0], span[1])
            samples.append(arr)
        return jnp.concatenate(samples, axis=0)

    @staticmethod
    def sample_hypercube(key, ranges, n_samples=1, method="uniform"):
        """
        Samples points from a multi-dimensional hypercube using various methods.

        This function generates `n_samples` points within a D-dimensional box
        defined by the `ranges` dictionary. It supports standard uniform random
        sampling as well as several quasi-random, space-filling sequences from
        `scipy.stats.qmc` that are often superior for exploring parameter spaces.

        Args:
            key: A JAX PRNG key for reproducibility.
            ranges (dict): A dictionary mapping parameter names to their
                           `(min, max)` range.
            n_samples (int): The total number of sample points to generate.
            method (str): The sampling strategy. One of:
                - "uniform": Standard pseudo-random sampling.
                - "sobol": Sobol sequence (quasi-random).
                - "halton": Halton sequence (quasi-random).
                - "latin_hypercube": Latin Hypercube Sampling (stratified).

        Returns:
            A dictionary mapping parameter names to JAX arrays of shape
            `(n_samples,)`.

        Raises:
            ValueError: If an unknown sampling method is specified.
        """
        param_names = list(ranges.keys())
        dim = len(param_names)

        lows = jnp.array([ranges[k][0] for k in param_names])
        highs = jnp.array([ranges[k][1] for k in param_names])

        # Generate points in the unit hypercube [0, 1]^D
        if method == "uniform":
            unit_points = random.uniform(key, shape=(n_samples, dim))
        elif method in ["sobol", "halton", "latin_hypercube"]:
            # Scipy's QMC samplers require a standard integer seed
            seed = int(random.randint(key, shape=(), minval=0, maxval=2**31 - 1))

            if method == "sobol":
                sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
            elif method == "halton":
                sampler = qmc.Halton(d=dim, scramble=True, seed=seed)
            elif method == "latin_hypercube":
                sampler = qmc.LatinHypercube(d=dim, seed=seed)

            unit_points = jnp.array(sampler.random(n_samples))
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Scale the unit points to the target hypercube
        scaled_points = lows + unit_points * (highs - lows)

        # Return the results as a dictionary
        return {name: scaled_points[:, i] for i, name in enumerate(param_names)}