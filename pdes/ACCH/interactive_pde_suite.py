# -*- coding: utf-8 -*-
"""
Interactive PDE Exploration and Comparison Suite.

This module provides the `InteractivePDESuite` class, a powerful tool for
real-time exploration of a Cahn-Hilliard/Allen-Cahn PDE system. It leverages
JAX's JIT compilation for extremely fast numerical solutions, wrapped in an
intuitive `ipywidgets` dashboard.

The suite allows users to:
1.  Interactively explore the PDE's parameter space using sliders and see
    the solution update in real-time.
2.  Compare the numerical PDE solution against a pre-trained PINN model to
    visually assess the model's accuracy across different physical parameters.
"""

# ==============================================================================
# 1. Imports
# ==============================================================================

import time
from functools import partial

import diffrax as dfx
import ipywidgets as widgets
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from jax import config, jit
from matplotlib.lines import Line2D


# ==============================================================================
# 2. Interactive PDE Suite Class
# ==============================================================================

class InteractivePDESuite:
    """
    A unified class for fast, JIT-compiled PDE solving and model comparison.

    This class encapsulates a `diffrax`-based numerical solver for a dimensionless
    PDE system. The core solver is JIT-compiled at initialization for maximum
    performance, enabling real-time interactive dashboards.
    """

    def __init__(self, base_params):
        """
        Initializes the interactive suite.

        Args:
            base_params (dict): A dictionary of the base physical parameters
                                and simulation settings for the PDE.
        """
        print("Initializing InteractivePDESuite...")
        jax.config.update("jax_enable_x64", True)

        self.base_params = base_params
        for key, value in base_params.items():
            setattr(self, key, value)

        # --- Pre-compute dimensionless grids and constants ---
        self.x_range_nd = (self.x_range[0] / self.l_0, self.x_range[1] / self.l_0)
        self.t_range_nd = (self.t_range[0] / self.t_0, self.t_range[1] / self.t_0)
        self.xs_nd = jnp.linspace(self.x_range_nd[0], self.x_range_nd[1], self.nx)
        self.ts_nd = jnp.linspace(self.t_range_nd[0], self.t_range_nd[1], self.nt)
        self.dx_nd = self.xs_nd[1] - self.xs_nd[0]
        self.c_diff = self.c_se - self.c_le
        self.c_bc_left_nd = (1.0 - self.c_le) / self.c_diff
        self.c_bc_right_nd = (0.0 - self.c_le) / self.c_diff

        self.solve_compiled = jit(self._solve_internal)
        print("Solver has been JIT-compiled. Ready for interactive use.")

    @staticmethod
    def _map_span(u, src, tgt):
        """Linearly maps an array of values from a source range to a target range."""
        a, b = src
        c, d = tgt
        if a == b:
            return jnp.full_like(u, (c + d) / 2.0)
        return (u - a) * (d - c) / (b - a) + c

    # ==========================================================================
    # Core PDE & Numerical Methods
    # ==========================================================================

    @staticmethod
    @jit
    def h(phi_nd):
        """Auxiliary function h(phi) from the PDE definition."""
        return -2 * phi_nd**3 + 3 * phi_nd**2

    @staticmethod
    @jit
    def h_p(phi_nd):
        """First derivative of h(phi) with respect to phi."""
        return -6 * phi_nd**2 + 6 * phi_nd

    @staticmethod
    @jit
    def g_p(phi_nd):
        """Derivative of the double-well potential g(phi)."""
        return 2 * phi_nd * (1 - phi_nd) * (2 * phi_nd - 1)

    @partial(jit, static_argnums=(0,))
    def laplacian_nd(self, f_nd):
        """Computes the 1D Laplacian using a central finite difference scheme."""
        interior = (f_nd[2:] - 2 * f_nd[1:-1] + f_nd[:-2]) / self.dx_nd**2
        return jnp.zeros_like(f_nd).at[1:-1].set(interior)

    @partial(jit, static_argnums=(0,))
    def rhs_nd(self, ts_nd, y_nd, p):
        """Defines the right-hand-side of the dimensionless PDE system."""
        P_CH, P_AC1, P_AC2, P_AC3 = p["P_CH"], p["P_AC1"], p["P_AC2"], p["P_AC3"]
        c_nd, phi_nd = jnp.split(y_nd, 2)
        phi_nd = phi_nd.at[0].set(1.0).at[-1].set(0.0)
        c_nd = c_nd.at[0].set(self.c_bc_left_nd).at[-1].set(self.c_bc_right_nd)
        h_phi = self.h(phi_nd)
        dc_dt_nd = P_CH * (self.laplacian_nd(c_nd) - self.laplacian_nd(h_phi))
        dphi_dt_nd = (
            P_AC1 * (c_nd - h_phi) * self.h_p(phi_nd)
            + P_AC2 * self.g_p(phi_nd)
            + P_AC3 * self.laplacian_nd(phi_nd)
        )
        bc_idx = jnp.array([0, -1])
        dc_dt_nd = dc_dt_nd.at[bc_idx].set(0.0)
        dphi_dt_nd = dphi_dt_nd.at[bc_idx].set(0.0)
        return jnp.concatenate([dc_dt_nd, dphi_dt_nd])

    def _solve_internal(self, L, M, alpha_phi, omega_phi):
        """The internal, JIT-compiled core solver function."""
        p = {
            "P_CH": (2 * self.A * M * self.t_0) / (self.l_0**2),
            "P_AC1": 2 * self.A * L * self.t_0 * self.c_diff**2,
            "P_AC2": L * omega_phi * self.t_0,
            "P_AC3": (L * alpha_phi * self.t_0) / (self.l_0**2)
        }
        K = self.l_0 * jnp.sqrt(omega_phi / (2 * alpha_phi))
        phi0_nd = 0.5 * (1.0 - jnp.tanh(K * self.xs_nd))
        c0_nd = (self.c_se * self.h(phi0_nd) - self.c_le) / self.c_diff
        y0_nd = jnp.concatenate([c0_nd, phi0_nd])
        term = dfx.ODETerm(self.rhs_nd)
        solver = dfx.Kvaerno5()
        controller = dfx.PIDController(rtol=1e-4, atol=1e-5)
        saveat = dfx.SaveAt(ts=self.ts_nd)
        sol = dfx.diffeqsolve(
            term, solver, t0=self.ts_nd[0], t1=self.ts_nd[-1], dt0=None,
            y0=y0_nd, args=p, saveat=saveat, stepsize_controller=controller
        )
        return sol.ys

    # ==========================================================================
    # Utility & Plotting Methods
    # ==========================================================================

    def to_phys(self, y_nd_sol):
        """Converts a dimensionless solution array to physical units."""
        c_nd_sol = y_nd_sol[:, :self.nx]
        phi_nd_sol = y_nd_sol[:, self.nx:]
        c_phys = c_nd_sol * self.c_diff + self.c_le
        return {"c": c_phys, "phi": phi_nd_sol}

    def plot_dashboard(self, sol, params):
        """Generates a 2x2 dashboard of heatmaps and profile plots."""
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        x_phys = np.linspace(self.x_range[0], self.x_range[1], self.nx)
        t_phys = np.linspace(self.t_range[0], self.t_range[1], self.nt)
        phi, c = sol["phi"], sol["c"]
        extent = [x_phys.min(), x_phys.max(), t_phys.min(), t_phys.max()]
        im1 = ax[0, 0].imshow(phi, aspect='auto', origin='lower', extent=extent, cmap='viridis')
        fig.colorbar(im1, ax=ax[0, 0]); ax[0, 0].set_title('Heatmap: Phase Field ($\\phi$)')
        im2 = ax[0, 1].imshow(c, aspect='auto', origin='lower', extent=extent, cmap='inferno')
        fig.colorbar(im2, ax=ax[0, 1]); ax[0, 1].set_title('Heatmap: Concentration (c)')
        time_indices = np.linspace(0, self.nt - 1, 6, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, 6))
        for i, t_idx in enumerate(time_indices):
            ax[1, 0].plot(x_phys, phi[t_idx, :], color=colors[i], label=f't={t_phys[t_idx]:.1e}')
            ax[1, 1].plot(x_phys, c[t_idx, :], color=colors[i])
        ax[1, 0].set_title('Profiles: Phase Field ($\\phi$)'); ax[1, 0].legend()
        ax[1, 1].set_title('Profiles: Concentration (c)')
        L, M, alpha_phi, omega_phi = params
        fig.suptitle(f"L={L:.2e}, M={M:.2e}, α_φ={alpha_phi:.2e}, ω_φ={omega_phi:.2e}", fontsize=16)
        plt.show()

    # ==========================================================================
    # Interactive UI Methods
    # ==========================================================================

    def _create_ui(self, run_function, span_pde):
        """A generic helper to create an ipywidgets UI for a given function."""
        param_map = {
            'L': 'L (Mobility)', 'M': 'M (Diffusivity)',
            'alpha_phi': 'α_φ (Gradient Energy)', 'omega_phi': 'ω_φ (Potential Height)'
        }
        style = {'description_width': 'initial'}
        sliders = {}
        for key, desc in param_map.items():
            if key in span_pde:
                min_val, max_val = span_pde[key]
                sliders[key] = widgets.FloatLogSlider(
                    value=self.base_params[key], base=10,
                    min=np.log10(min_val), max=np.log10(max_val), step=0.05,
                    description=desc, readout_format='.2e', style=style,
                    continuous_update=False
                )
        if not sliders:
            print("No variable parameters specified. Running single simulation.")
            run_function(); return
        controls = widgets.VBox([widgets.HBox(list(sliders.values())[i:i+2]) for i in range(0, len(sliders), 2)])
        output_widget = widgets.Output(layout={'min_height': '850px'})
        def update_handler(**kwargs):
            with output_widget:
                output_widget.clear_output(wait=True); run_function(**kwargs)
        display(controls, output_widget)
        widgets.interactive_output(update_handler, sliders)
        update_handler(**{key: slider.value for key, slider in sliders.items()})

    def explore(self, span_pde=None):
        """
        Launches an interactive dashboard to explore the PDE parameter space.

        This method provides a user-friendly interface with sliders to change
        physical parameters and immediately see the impact on the PDE solution,
        visualized as a comprehensive 2x2 dashboard.

        Args:
            span_pde (dict, optional): A dictionary mapping parameter names
                                       ('L', 'M', etc.) to their (min, max)
                                       range for the interactive sliders.
        """
        span_pde = span_pde or {}
        def run_and_plot(**kwargs):
            current_params = {**self.base_params, **kwargs}
            L, M, alpha_phi, omega_phi = (current_params[k] for k in ['L', 'M', 'alpha_phi', 'omega_phi'])
            print("Solving PDE...")
            start_time = time.time()
            y_nd_sol = self.solve_compiled(L, M, alpha_phi, omega_phi)
            y_nd_sol.block_until_ready()
            print(f"PDE solved in {time.time() - start_time:.2f} seconds.")
            solution_phys = self.to_phys(y_nd_sol)
            self.plot_dashboard(solution_phys, (L, M, alpha_phi, omega_phi))
        self._create_ui(run_and_plot, span_pde)

    def create_interactive_comparison_plot(self, model, span_pde, span_model, num_frames=5,
                                           prediction_color='red'):
        """
        Launches an interactive dashboard to compare the PDE solver against a PINN model.

        This powerful visualization tool allows for real-time, qualitative
        assessment of a trained PINN's generalization capabilities. By adjusting
        the physical parameters with sliders, one can instantly see how well the
        model's predictions (dashed lines) match the numerical ground truth
        (solid lines).

        Args:
            model: A trained PINN model instance with a `validation` method.
            span_pde (dict): Parameter ranges for the interactive sliders.
            span_model (dict): The model's normalized input spans.
            num_frames (int): Number of time snapshots to plot.
            prediction_color (str or None): Controls the prediction line color.
                - 'red' (default) or any color string: Plots all predictions in that color.
                - 'match': Plots each prediction line with the same color as its
                           corresponding ground truth line for direct comparison.
                - None: Disables plotting of model predictions entirely.
        """
        def run_and_plot_comparison(**kwargs):
            current_params = {**self.base_params, **kwargs}
            L, M, alpha_phi, omega_phi = (current_params[k] for k in ['L', 'M', 'alpha_phi', 'omega_phi'])
            print("Solving PDE for ground truth...")
            start_time = time.time()
            y_nd_sol = self.solve_compiled(L, M, alpha_phi, omega_phi)
            y_nd_sol.block_until_ready()
            print(f"PDE solved in {time.time() - start_time:.2f} seconds.")
            c_nd_true = y_nd_sol[:, :self.nx]
            phi_nd_true = y_nd_sol[:, self.nx:]

            fig, ax = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
            time_indices = np.linspace(0, self.nt - 1, num_frames, dtype=int)
            colors = plt.cm.viridis(np.linspace(0, 1, num_frames))

            if prediction_color is not None:
                print("Generating model predictions...")
                start_time = time.time()
                x_model_grid = np.linspace(span_model['x'][0], span_model['x'][1], self.nx)
                t_model_grid = np.linspace(span_model['t'][0], span_model['t'][1], self.nt)
                scalar_names = sorted([k for k in model.inp_idx if k not in ['x', 't']], key=model.inp_idx.get)
                ordered_scalar_args = [
                    self._map_span(current_params[name], span_pde[name], span_model[name])
                    for name in scalar_names
                ]
                pred_results = model.validation(x_model_grid, t_model_grid, *ordered_scalar_args)
                c_pred, phi_pred = pred_results['c'], pred_results['phi']
                print(f"Model predictions generated in {time.time() - start_time:.2f} seconds.")

            for i, t_idx in enumerate(time_indices):
                label = f"t'={self.ts_nd[t_idx]:.2f}"
                truth_color = colors[i]
                ax[0].plot(self.xs_nd, c_nd_true[t_idx, :], color=truth_color, ls='-', lw=3, label=label)
                ax[1].plot(self.xs_nd, phi_nd_true[t_idx, :], color=truth_color, ls='-', lw=3, label=label)

                if prediction_color is not None:
                    pred_line_color = truth_color if prediction_color == 'match' else prediction_color
                    ax[0].plot(self.xs_nd, c_pred[t_idx, :], color=pred_line_color, ls='--', lw=2.5)
                    ax[1].plot(self.xs_nd, phi_pred[t_idx, :], color=pred_line_color, ls='--', lw=2.5)

            legend_elements = [Line2D([0], [0], color='black', lw=3, ls='-', label='PDE Truth')]
            if prediction_color is not None:
                legend_pred_color = 'black' if prediction_color == 'match' else prediction_color
                legend_elements.append(
                    Line2D([0], [0], color=legend_pred_color, lw=2.5, ls='--', label='Model Pred.')
                )
            ax[0].set_title("Dimensionless Concentration (c')", fontsize=16)
            ax[0].set_xlabel("Dimensionless Position (x')", fontsize=12)
            ax[0].legend(handles=legend_elements)
            ax[1].set_title("Dimensionless Phase Field ($\\phi'$)", fontsize=16)
            ax[1].set_xlabel("Dimensionless Position (x')", fontsize=12)
            ax[1].legend(handles=legend_elements)
            fig.suptitle(f"L={L:.2e}, M={M:.2e}, α_φ={alpha_phi:.2e}, ω_φ={omega_phi:.2e}", fontsize=18)
            plt.show()

        self._create_ui(run_and_plot_comparison, span_pde)