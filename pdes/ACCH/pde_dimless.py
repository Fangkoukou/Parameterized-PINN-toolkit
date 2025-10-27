"""
Dimensionless PDE Solver for Cahn-Hilliard and Allen-Cahn Systems.

This module provides the `PDE_dimless` class, a tool for solving a coupled
system of Cahn-Hilliard and Allen-Cahn equations. It operates by first
converting a set of physical parameters into a dimensionless form, which
stabilizes the numerical solution process.

The core functionality includes:
- A high-performance numerical solver built on `diffrax`.
- Methods for defining initial/boundary conditions and PDE-specific terms.
- A JIT-compiled, parallel data generation pipeline using `vmap` for creating
  training datasets efficiently.
"""

# ==============================================================================
# 1. Imports
# ==============================================================================

from contextlib import nullcontext

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import config, random, vmap

from pinn_toolkit.util import get_len, map_span

# JAX configuration for high precision
config.update("jax_enable_x64", True)

# ==============================================================================
# 2. PDE Solver Class
# ==============================================================================


class PDE_dimless:
    """
    Solves the Cahn-Hilliard/Allen-Cahn PDE system in dimensionless form.

    This class takes physical parameters, converts them to a dimensionless
    system, solves the system on a 1D grid over time using a numerical ODE
    solver, and provides utilities to convert the results back to physical units.

    Attributes:
        Many physical and dimensionless parameters are set dynamically in `__init__`.
        xs_nd (jnp.ndarray): The dimensionless 1D spatial grid.
        ts_nd (jnp.ndarray): The dimensionless time points for the solution.
        dx_nd (float): The spacing of the dimensionless spatial grid.
    """

    def __init__(self, params):
        """
        Initializes the PDE solver with physical parameters.

        This constructor stores all provided physical parameters and pre-computes
        the dimensionless grids and boundary conditions required for the solver.

        Args:
            params (dict): A dictionary of physical parameters (e.g., 'A', 'M', 'L')
                           and simulation settings (e.g., 'nx', 'nt', 'x_range').
        """
        # --- Store all physical parameters from the input dictionary ---
        for key, value in params.items():
            setattr(self, key, value)

        # --- Pre-compute dimensionless grids and constants ---
        self.x_range_nd = (self.x_range[0] / self.l_0, self.x_range[1] / self.l_0)
        self.t_range_nd = (self.t_range[0] / self.t_0, self.t_range[1] / self.t_0)
        self.xs_nd = jnp.linspace(self.x_range_nd[0], self.x_range_nd[1], self.nx)
        self.ts_nd = jnp.linspace(self.t_range_nd[0], self.t_range_nd[1], self.nt)
        self.dx_nd = self.xs_nd[1] - self.xs_nd[0]

        # --- Pre-compute dimensionless boundary conditions for 'c' ---
        self.c_diff = self.c_se - self.c_le
        self.c_bc_left_nd = (1.0 - self.c_le) / self.c_diff
        self.c_bc_right_nd = (0.0 - self.c_le) / self.c_diff

    # ==========================================================================
    # 3. Utility Methods
    # ==========================================================================

    def show(self, **overrides):
        """
        Prints a formatted table of physical and dimensionless parameters.

        This is a convenience method for inspecting the current state of the
        solver's parameters.

        Args:
            **overrides: Allows temporary overriding of parameters (e.g., L=0.1)
                         to see their effect on the dimensionless values.
        """
        phys_params = [
            "alpha_phi", "omega_phi", "M", "L", "A", "c_se", "c_le",
            "x_range", "t_range", "l_0", "t_0"
        ]
        dimless_params = [
            "P_CH", "P_AC1", "P_AC2", "P_AC3", "x_range_nd", "t_range_nd"
        ]

        # Pad lists to the same length for zipping
        max_len = max(len(phys_params), len(dimless_params))
        phys_params += [""] * (max_len - len(phys_params))
        dimless_params += [""] * (max_len - len(dimless_params))

        print(f"{'Physical Parameter':<20}{'Value':<25}|| {'Dimless Parameter':<20}{'Value':<25}")
        print("=" * 95)

        def fmt(val):
            """Helper to format values nicely for printing."""
            if isinstance(val, (int, float)): return f"{val:.5g}"
            if isinstance(val, (list, tuple)): return str([f"{v:.5g}" for v in val])
            return str(val)

        for p_attr, d_attr in zip(phys_params, dimless_params):
            p_val_str = fmt(getattr(self, p_attr, "")) if p_attr else ""
            d_val = getattr(self, d_attr, "") if d_attr else ""
            if callable(d_val):
                try: d_val = d_val(**overrides)
                except TypeError: d_val = "<func>"
            d_val_str = fmt(d_val)
            print(f"{p_attr:<20}{p_val_str:<25}|| {d_attr:<20}{d_val_str:<25}")

    def dimless_params(self, **overrides):
        """Returns a dictionary of the core dimensionless parameters."""
        keys = ["P_CH", "P_AC1", "P_AC2", "P_AC3"]
        out = {}
        for k in keys:
            val = getattr(self, k)
            if callable(val):
                val = val(**overrides)
            out[k] = val
        return out

    # ==========================================================================
    # 4. Dimensionless Parameters & Auxiliary Functions
    # ==========================================================================

    @staticmethod
    def h(phi_nd):
        """Auxiliary function h(phi) from the PDE definition."""
        return -2 * phi_nd**3 + 3 * phi_nd**2

    @staticmethod
    def h_p(phi_nd):
        """First derivative of h(phi) with respect to phi."""
        return -6 * phi_nd**2 + 6 * phi_nd

    @staticmethod
    def h_pp(phi_nd):
        """Second derivative of h(phi) with respect to phi."""
        return -12.0 * phi_nd + 6.0

    @staticmethod
    def g_p(phi_nd):
        """Derivative of the double-well potential g(phi)."""
        return 2 * phi_nd * (1 - phi_nd) * (2 * phi_nd - 1)

    def P_CH(self, **overrides):
        """Dimensionless parameter for the Cahn-Hilliard equation."""
        M = overrides.get('M', self.M)
        return (2 * self.A * M * self.t_0) / (self.l_0**2)

    def P_AC1(self, **overrides):
        """First dimensionless parameter for the Allen-Cahn equation."""
        L = overrides.get('L', self.L)
        return 2 * self.A * L * self.t_0 * self.c_diff**2

    def P_AC2(self, **overrides):
        """Second dimensionless parameter for the Allen-Cahn equation."""
        L = overrides.get('L', self.L)
        omega_phi = overrides.get('omega_phi', self.omega_phi)
        return L * omega_phi * self.t_0

    def P_AC3(self, **overrides):
        """Third dimensionless parameter for the Allen-Cahn equation."""
        L = overrides.get('L', self.L)
        alpha_phi = overrides.get('alpha_phi', self.alpha_phi)
        return (L * alpha_phi * self.t_0) / (self.l_0**2)

    def K_nd(self, **overrides):
        """Dimensionless width parameter for the initial phi profile."""
        omega_phi = overrides.get('omega_phi', self.omega_phi)
        alpha_phi = overrides.get('alpha_phi', self.alpha_phi)
        return self.l_0 * jnp.sqrt(omega_phi / (2 * alpha_phi))

    # ==========================================================================
    # 5. Dimensionless Initial & Boundary Conditions
    # ==========================================================================

    def phi_ic_nd(self, xs_nd, ts_nd, **overrides):
        """Defines the initial condition for phi' as a hyperbolic tangent."""
        x_init_nd = overrides.get("x_init", 0.0) / self.l_0
        xd_nd = xs_nd - x_init_nd
        K = self.K_nd(**overrides)
        return 0.5 * (1.0 - jnp.tanh(K * xd_nd))

    def c_ic_nd(self, xs_nd, ts_nd, **overrides):
        """Defines the initial condition for c' based on the initial phi'."""
        phi_nd = self.phi_ic_nd(xs_nd, ts_nd, **overrides)
        h_phi = self.h(phi_nd)
        return (self.c_se * h_phi - self.c_le) / self.c_diff

    def phi_bc_nd(self, xs_nd, ts_nd):
        """Defines the Dirichlet boundary condition for phi'."""
        return jnp.where(xs_nd <= self.x_range_nd[0], 1.0, 0.0)

    def c_bc_nd(self, xs_nd, ts_nd):
        """Defines the Dirichlet boundary condition for c'."""
        return jnp.where(xs_nd <= self.x_range_nd[0], self.c_bc_left_nd, self.c_bc_right_nd)

    # ==========================================================================
    # 6. Numerical Core
    # ==========================================================================

    def laplacian_nd(self, f_nd):
        """Computes the 1D Laplacian using a central finite difference scheme."""
        lap = jnp.zeros_like(f_nd)
        interior = (f_nd[2:] - 2 * f_nd[1:-1] + f_nd[:-2]) / self.dx_nd**2
        return lap.at[1:-1].set(interior)

    def rhs_nd(self, ts_nd, y_nd, p):
        """
        Defines the right-hand-side of the dimensionless PDE system for the ODE solver.

        This function calculates the time derivatives (dc'/dt', dphi'/dt') that
        drive the simulation forward.

        Args:
            ts_nd (float): The current dimensionless time.
            y_nd (jnp.ndarray): The current state vector [c', phi'].
            p (dict): A dictionary of the dimensionless parameters (P_CH, etc.).

        Returns:
            jnp.ndarray: The time derivatives [dc'/dt', dphi'/dt'].
        """
        # Unpack the state vector into its components
        c_nd, phi_nd = jnp.split(y_nd, 2)

        # Enforce Dirichlet boundary conditions on the state vector
        phi_nd = phi_nd.at[0].set(self.phi_bc_nd(self.xs_nd[0], ts_nd))
        phi_nd = phi_nd.at[-1].set(self.phi_bc_nd(self.xs_nd[-1], ts_nd))
        c_nd = c_nd.at[0].set(self.c_bc_nd(self.xs_nd[0], ts_nd))
        c_nd = c_nd.at[-1].set(self.c_bc_nd(self.xs_nd[-1], ts_nd))

        # Calculate intermediate terms and Laplacians
        h_phi = self.h(phi_nd)
        lap_c_nd = self.laplacian_nd(c_nd)
        lap_phi_nd = self.laplacian_nd(phi_nd)
        lap_h_phi = self.laplacian_nd(h_phi)

        # --- Dimensionless Cahn-Hilliard Equation ---
        dc_dt_nd = p["P_CH"] * (lap_c_nd - lap_h_phi)

        # --- Dimensionless Allen-Cahn Equation ---
        reaction_term = p["P_AC1"] * (c_nd - h_phi) * self.h_p(phi_nd)
        potential_term = p["P_AC2"] * self.g_p(phi_nd)
        gradient_term = p["P_AC3"] * lap_phi_nd
        dphi_dt_nd = reaction_term + potential_term + gradient_term

        # For Dirichlet BCs, the time derivative at the boundary must be zero
        bc_idx = jnp.array([0, -1])
        dc_dt_nd = dc_dt_nd.at[bc_idx].set(0.0)
        dphi_dt_nd = dphi_dt_nd.at[bc_idx].set(0.0)

        return jnp.concatenate([dc_dt_nd, dphi_dt_nd])

    # ==========================================================================
    # 7. Solver and Data Generation
    # ==========================================================================

    def solve(self, force_cpu=False, **overrides):
        """
        Solves the dimensionless PDE system.

        Args:
            force_cpu (bool): If True, forces the computation onto the CPU,
                              which can be useful for very stiff problems.
            **overrides: Physical parameters to override for this specific run
                         (e.g., L=0.1, x_init=5.0).

        Returns:
            dict: A dictionary containing the dimensionless solution arrays
                  for 'x', 't', 'phi', and 'c'.
        """
        # 1. Calculate dimensionless parameters for this specific run
        p = self.dimless_params(**overrides)

        # 2. Set up the initial state vector y0 = [c0, phi0]
        phi0_nd = self.phi_ic_nd(self.xs_nd, 0.0, **overrides)
        c0_nd = self.c_ic_nd(self.xs_nd, 0.0, **overrides)
        y0_nd = jnp.concatenate([c0_nd, phi0_nd])

        # 3. Define the ODE problem and solver using diffrax
        term = dfx.ODETerm(self.rhs_nd)
        solver = dfx.Kvaerno5()  # A good solver for stiff problems
        controller = dfx.PIDController(rtol=1e-7, atol=1e-9)
        saveat = dfx.SaveAt(ts=self.ts_nd)

        # 4. Run the solver
        context = jax.default_device(jax.devices("cpu")[0]) if force_cpu else nullcontext()
        with context:
            sol = dfx.diffeqsolve(
                term, solver, t0=self.ts_nd[0], t1=self.ts_nd[-1], dt0=1e-5,
                y0=y0_nd, args=p, saveat=saveat,
                stepsize_controller=controller, max_steps=400000,
            )

        # 5. Unpack and return the results
        c_nd_sol = sol.ys[:, :self.nx]
        phi_nd_sol = sol.ys[:, self.nx:]
        return {"x": self.xs_nd, "t": self.ts_nd, "phi": phi_nd_sol, "c": c_nd_sol}

    def to_phys(self, result_nd):
        """Converts a dimensionless result dictionary to physical units."""
        return {
            "x": map_span(result_nd['x'], self.x_range_nd, self.x_range),
            "t": map_span(result_nd['t'], self.t_range_nd, self.t_range),
            "phi": result_nd['phi'],
            "c": result_nd['c'] * self.c_diff + self.c_le
        }

    @staticmethod
    @eqx.filter_jit
    def _generate_single_jit(pde_solver_instance, key, p):
        """
        A JIT-compiled static method to run one full simulation and format the data.

        `eqx.filter_jit` automatically treats the first argument (the class instance)
        as static, meaning its methods can be called but its data is not traced.
        This allows the entire simulation to be compiled for a given set of
        overridden physical parameters `p`.
        """
        # Run the solver on the default device (e.g., GPU)
        sol = pde_solver_instance.solve(**p)

        # Format the solution into a flat "training data" format
        xs_nd, ts_nd = jnp.meshgrid(sol['x'], sol['t'])
        train_data = {
            'x': xs_nd.ravel(), 't': ts_nd.ravel(),
            'phi': sol['phi'].ravel(), 'c': sol['c'].ravel()
        }
        return sol, train_data

    def generate_training_data(self, key, params, num_train):
        """
        Generates a training dataset by running multiple simulations in parallel.

        This function leverages `jax.vmap` to map the JIT-compiled single-simulation
        runner over a batch of varying physical parameters.

        Args:
            key: A JAX PRNG key.
            params (dict): A PyTree where each leaf is an array of physical
                           parameters to be varied. The length of the arrays
                           determines the number of simulations.
            num_train (int): (Not used in this implementation, but kept for API
                             consistency).

        Returns:
            A tuple of (solutions, training_data), where each is a PyTree
            containing the batched results from all simulations.
        """
        num_simulations = get_len(params)
        keys = random.split(key, num_simulations)

        # Vmap the JIT-compiled solver over the keys and parameters
        return vmap(PDE_dimless._generate_single_jit, in_axes=(None, 0, 0))(self, keys, params)