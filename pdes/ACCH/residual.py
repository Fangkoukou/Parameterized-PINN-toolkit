"""
PINN Residual and Loss Computation Module.

This module defines the `Residual` class, which is responsible for calculating
the residuals of a PINN model against various physical constraints:
- Initial Conditions (IC)
- Boundary Conditions (BC)
- The governing Partial Differential Equation (PDE)
- Supervised data points

It also includes methods for computing the total loss and for dynamically
weighting loss components using the Neural Tangent Kernel (NTK).
"""

# ==============================================================================
# 1. Imports
# ==============================================================================

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import grad, lax, vmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_map

# ==============================================================================
# 2. Residual Class
# ==============================================================================


class Residual:
    """
    Computes residuals for PINN loss functions and their NTK-based weights.

    This class encapsulates the logic for evaluating how well a neural network
    satisfies the initial conditions, boundary conditions, and the governing
    PDEs (Allen-Cahn and Cahn-Hilliard in this case). It handles the necessary
    coordinate transformations (denormalization) between the model's normalized
    space and the physical problem space.

    Attributes:
        span_pde (dict): Dictionary of (min, max) spans for variables in the
                         physical (or dimensionless) PDE space.
        span_model (dict): Dictionary of (min, max) spans for variables in the
                           model's normalized input space (typically [-1, 1]).
        pdekernel: An object containing the analytical PDE functions.
        deriv: A derivative utility object for computing gradients of the NN.
    """

    def __init__(self, span_pde, span_model, pdekernel, derivative):
        """Initializes the Residual computer."""
        self.span_pde = span_pde
        self.span_model = span_model
        self.pdekernel = pdekernel
        self.deriv = derivative

    @staticmethod
    def _map_span(u, src, tgt):
        """Linearly maps a value from a source span to a target span."""
        a, b = src
        c, d = tgt
        return c + (u - a) * (d - c) / (b - a)

    def denorm(self, D, keys_to_denorm=None):
        """
        Denormalizes values from the model's space to the physical PDE space.

        This method takes a dictionary of inputs that are normalized (e.g., in
        the range [-1, 1]) for the model and converts specified keys back to
        their physical scales as defined by `self.span_pde`.

        Args:
            D (dict): The input dictionary with values in the model's space.
            keys_to_denorm (list, optional): A specific list of keys to
                                             denormalize. If None, all keys in D
                                             are considered.

        Returns:
            dict: A new dictionary with the specified values denormalized.
        """
        out_dict = D.copy()
        keys_to_process = keys_to_denorm if keys_to_denorm is not None else D.keys()

        for key in keys_to_process:
            # Only transform keys that have a defined mapping in both spans
            if key in self.span_pde and key in self.span_model and key in D:
                out_dict[key] = self._map_span(
                    D[key], self.span_model[key], self.span_pde[key]
                )
        return out_dict

    def res_ic(self, model, inp_model):
        """Computes the residual for the initial conditions (IC)."""
        # Get model prediction in normalized space
        pred = model.predict(inp_model)
        phi_pred, c_pred = pred['phi'], pred['c']

        # Get ground truth by denormalizing inputs and calling the analytical IC function
        inp_pde = self.denorm(inp_model)
        phi_true = self.pdekernel.phi_ic_nd(inp_pde['x'], inp_pde['t'], **inp_pde)
        c_true = self.pdekernel.c_ic_nd(inp_pde['x'], inp_pde['t'], **inp_pde)

        # The residual is the difference between prediction and ground truth
        res = jnp.stack([phi_pred - phi_true, c_pred - c_true], axis=0).ravel()
        return {'ic': res}

    def res_bc(self, model, inp_model):
        """Computes the residual for the boundary conditions (BC)."""
        pred = model.predict(inp_model)
        phi_pred, c_pred = pred['phi'], pred['c']

        inp_pde = self.denorm(inp_model)
        phi_true = self.pdekernel.phi_bc_nd(inp_pde['x'], inp_pde['t'])
        c_true = self.pdekernel.c_bc_nd(inp_pde['x'], inp_pde['t'])

        res = jnp.stack([phi_pred - phi_true, c_pred - c_true], axis=0).ravel()
        return {'bc': res}

    def res_pde(self, model, inp_model):
        """Computes the residual for the governing PDEs (Allen-Cahn & Cahn-Hilliard)."""
        # Get model prediction for the output variables
        pred = model.predict(inp_model)
        phi_pred, c_pred = pred['phi'], pred['c']

        # Compute required derivatives of the NN outputs w.r.t. its inputs
        sorted_inp_names = sorted(self.deriv.inp_idx.keys(), key=self.deriv.inp_idx.get)
        ordered_args = [inp_model[name] for name in sorted_inp_names]
        derivs = self.deriv.evaluate(
            model, *ordered_args, function_names=['phi_t', 'phi_x', 'phi_2x', 'c_t', 'c_2x']
        )

        # Compute intermediate terms required by the PDEs
        h = self.pdekernel.h(phi_pred)
        h_p = self.pdekernel.h_p(phi_pred)
        h_xx = self.pdekernel.h_pp(phi_pred) * derivs['phi_x']**2 + h_p * derivs['phi_2x']
        g_p = self.pdekernel.g_p(phi_pred)

        # Denormalize inputs to get physical parameters for the PDE coefficients
        inp_pde = self.denorm(inp_model)

        # Assemble the final PDE residuals. The residual is zero when the PDE is satisfied.
        res_phi = (
            derivs['phi_t']
            - self.pdekernel.P_AC1(**inp_pde) * (c_pred - h) * h_p
            - self.pdekernel.P_AC2(**inp_pde) * g_p
            - self.pdekernel.P_AC3(**inp_pde) * derivs['phi_2x']
        )
        res_c = derivs['c_t'] - (derivs['c_2x'] - h_xx) * self.pdekernel.P_CH(**inp_pde)

        return {'ac': res_phi.ravel(), 'ch': res_c.ravel()}

    def res_data(self, model, inp_model):
        """Computes the residual for supervised data points."""
        if not inp_model:  # Handle case with no data points
            return {'data': jnp.array(0.0)}

        pred = model.predict(inp_model)
        phi_pred, c_pred = pred['phi'], pred['c']

        # The residual is the difference between prediction and the provided data values
        res = jnp.stack([phi_pred - inp_model['phi'], c_pred - inp_model['c']], axis=0).ravel()
        return {'data': res}

    def compute_loss(self, model, combined_inp):
        """
        Computes the total loss as the Mean Squared Error (MSE) of all residuals.

        Args:
            model (eqx.Module): The PINN model.
            combined_inp (dict): A dictionary containing inputs for each residual
                                 type, e.g., `{'ic': ic_points, 'bc': bc_points, ...}`.

        Returns:
            dict: A dictionary mapping each residual name to its MSE loss.
        """
        residuals = {}
        residuals.update(self.res_ic(model, combined_inp['ic']))
        residuals.update(self.res_bc(model, combined_inp['bc']))
        residuals.update(self.res_pde(model, combined_inp['colloc']))
        residuals.update(self.res_data(model, combined_inp['data']))
        return {k: jnp.mean(v**2) for k, v in residuals.items()}

    def ntk_residual_wrappers(self):
        """
        Creates NTK-compatible wrappers for each residual function.

        The NTK computation requires a function with a specific signature:
        `f(flat_params, reconstruct_fn, static_params, input_data)`.
        This method wraps each residual function to match this signature.

        Returns:
            dict: A dictionary of wrapped residual functions.
        """
        def wrap(fn, key):
            def wrapped(flat_p, recon, static, inp):
                # Reconstruct the model inside from flattened parameters
                model = eqx.combine(recon(flat_p), static)
                return fn(model, inp)[key]
            return wrapped

        return {
            'ic': wrap(self.res_ic, 'ic'),
            'bc': wrap(self.res_bc, 'bc'),
            'ac': wrap(self.res_pde, 'ac'),
            'ch': wrap(self.res_pde, 'ch'),
            'data': wrap(self.res_data, 'data'),
        }

    def compute_ntk_weights(self, model, inp, batch_size=128):
        """
        Computes NTK-based weights for balancing loss components.

        This method approximates the trace of the Neural Tangent Kernel for each
        residual type. The inverse of this trace can be used as a weight to
        balance the learning speed of different loss terms. The computation is
        batched to handle large datasets without memory overflow.

        Args:
            model (eqx.Module): The current PINN model.
            inp (dict): A dictionary of input points for all residual types.
            batch_size (int): The batch size for computing the NTK trace.

        Returns:
            dict: A dictionary of computed weights for each loss component.
        """
        eps = 1e-11
        params, static = eqx.partition(model, eqx.is_inexact_array)
        flat_p, recon = ravel_pytree(params)
        res_fns = self.ntk_residual_wrappers()
        inps = {
            'ic': inp['ic'], 'bc': inp['bc'], 'ac': inp['colloc'],
            'ch': inp['colloc'], 'data': inp['data']
        }

        def batched_trace(fn, _inp):
            """Computes the trace of the NTK Jacobian in batches."""
            leaves = tree_leaves(_inp)
            if not leaves or leaves[0].shape[0] == 0:
                return 0.0, 0
            n = leaves[0].shape[0]

            # Function to compute the squared Frobenius norm of the Jacobian for a single point
            def single_point_trace(__inp_slice):
                local_fn = lambda p: fn(p, recon, static, __inp_slice)
                # J(p)^2 is the trace of J(p)^T @ J(p), which is the NTK
                return jnp.sum(jax.jacrev(local_fn)(flat_p) ** 2)

            batch_trace_fn = vmap(single_point_trace)

            # Truncate input to be divisible by batch_size for lax.scan
            num_full_batches = n // batch_size
            n_processed = num_full_batches * batch_size
            if n_processed == 0: return 0.0, 0 # Handle cases smaller than batch_size
            
            truncated_inp = tree_map(lambda x: x[:n_processed], _inp)
            batched_inp = tree_map(
                lambda x: x.reshape(num_full_batches, batch_size, *x.shape[1:]),
                truncated_inp
            )

            # Use lax.scan for efficient, JIT-compatible batch processing
            def scan_body(accumulated_total, one_batch):
                batch_total = jnp.sum(batch_trace_fn(one_batch))
                return accumulated_total + batch_total, None

            final_total, _ = lax.scan(scan_body, 0.0, batched_inp)
            return final_total, n_processed

        # Compute raw weights as n / Tr(NTK)
        raw_weight = {}
        total_trace_sum = 0
        for k, _inp in inps.items():
            total_trace, n = batched_trace(res_fns[k], _inp)
            raw_weight[k] = n / (total_trace + eps)
            total_trace_sum += total_trace / (n + eps)

        # Normalize weights by the average trace across all components
        return {k: u * total_trace_sum for k, u in raw_weight.items()}