"""
Module for generating parameter sets, running parallel PDE simulations,
and managing the resulting data with HDF5 files.
"""

# ==============================================================================
# --- 1. IMPORTS
# ==============================================================================
import time
import warnings
import jax
import jax.numpy as jnp
import os
import h5py
import numpy as np
import multiprocessing as mp
import concurrent.futures
from scipy.stats import qmc
from scipy.integrate import solve_ivp
from tqdm import tqdm # Use standard tqdm for script-based execution
from typing import Union, List, Set, Tuple, Dict, Any
from typing import Dict, Any, Optional, Union, List, Set
import os
import time
import tempfile
import shutil
import numpy as np
import h5py
import concurrent.futures
import multiprocessing as mp
from scipy.integrate import solve_ivp
from scipy.stats import qmc
from tqdm.notebook import tqdm
from typing import Dict, Any, List, Set

# --- Environment defaults to avoid BLAS/thread oversubscription in workers ---
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# Try to set spawn method (safe for scripts). Ignore if already set.
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass



# Define what functions are publicly accessible when importing with '*'
__all__ = [
    'generate_parameter_sets',
    'run_and_save_simulations',
    'load_h5',
    'solve_pde'
]


# ==============================================================================
# --- 2. HELPER FUNCTIONS
# ==============================================================================

def map_span(u: float, src: Tuple[float, float], tgt: Tuple[float, float]) -> float:
    """Maps a value from a source span to a target span."""
    a, b = src
    c, d = tgt
    if b == a:
        return (c + d) / 2
    if not a <= u <= b:
        warnings.warn(f"Input value {u} is outside the source range {src}", UserWarning)
    
    # Perform linear mapping
    return c + (u - a) * (d - c) / (b - a)

# Note: The `map_span_dict` function was not used in the final workflow,
# but it is kept here in case it's needed for other purposes.
def map_span_dict(d_src: Dict, span_src: Dict, span_tgt: Dict) -> Dict:
    """Maps dictionary values from a source range to a target range."""
    return {
        key: map_span(value, span_src[key], span_tgt[key])
        if key in span_src and key in span_tgt
        else value
        for key, value in d_src.items()
    }


# ==============================================================================
# --- 3. CORE SCIENTIFIC LOGIC
# ==============================================================================

def solve_pde(params):
    """
    Solves the dimensionless PDE system using NumPy and SciPy.
    Accepts a single dictionary containing all necessary parameters.
    Returns the solution and the time taken for the solve step.
    API and returned structure preserved.
    """
    start_time = time.perf_counter()

    # --- Extract and validate parameters ---
    A = params['A']
    M = params['M']
    L = params['L']
    nx = int(params['nx'])
    nt = int(params['nt'])
    x_range = params['x_range']
    t_range = params['t_range']
    c_se = params['c_se']
    c_le = params['c_le']
    omega_phi = params['omega_phi']
    alpha_phi = params['alpha_phi']
    l_0 = params['l_0']
    t_0 = params['t_0']
    x_init = params.get('x_init', 0.0)

    # --- Grids and basic constants ---
    xs_nd = np.linspace(x_range[0] / l_0, x_range[1] / l_0, nx)
    ts_nd = np.linspace(t_range[0] / t_0, t_range[1] / t_0, nt)
    dx_nd = xs_nd[1] - xs_nd[0]
    inv_dx2 = 1.0 / (dx_nd * dx_nd)

    c_diff = c_se - c_le
    c_bc_left_nd = (1.0 - c_le) / c_diff
    c_bc_right_nd = (0.0 - c_le) / c_diff

    # --- Dimensionless parameters ---
    P_CH = (2 * A * M * t_0) / (l_0**2)
    P_AC1 = 2 * A * L * t_0 * (c_diff**2)
    P_AC2 = L * omega_phi * t_0
    P_AC3 = (L * alpha_phi * t_0) / (l_0**2)
    K = l_0 * np.sqrt(omega_phi / (2 * alpha_phi))

    # --- Initial condition helpers ---
    def phi_ic_nd(x_nd):
        xd_nd = x_nd - x_init / l_0
        return 0.5 * (1.0 - np.tanh(K * xd_nd))

    def c_ic_nd(x_nd):
        phi = phi_ic_nd(x_nd)
        h = -2 * phi**3 + 3 * phi**2
        return (c_se * h - c_le) / c_diff

    # --- Preallocated reusable arrays for laplacians ---
    _lap_c = np.zeros(nx, dtype=float)
    _lap_phi = np.zeros(nx, dtype=float)
    _lap_h = np.zeros(nx, dtype=float)

    def laplacian_nd_inplace(f_nd, out):
        out.fill(0.0)
        out[1:-1] = (f_nd[2:] - 2.0 * f_nd[1:-1] + f_nd[:-2]) * inv_dx2
        return out

    # --- RHS for full-state (c and phi stacked) ---
    def rhs_nd(t, y):
        # y is length 2*nx: [c, phi]
        c_nd, phi_nd = np.split(y, 2)

        # enforce boundary values (same behavior as original)
        phi_nd[0], phi_nd[-1] = 1.0, 0.0
        c_nd[0], c_nd[-1] = c_bc_left_nd, c_bc_right_nd

        # nonlinear terms
        h_phi = -2.0 * phi_nd**3 + 3.0 * phi_nd**2
        dh_dphi = -6.0 * phi_nd**2 + 6.0 * phi_nd

        # laplacians (reuse buffers)
        lap_c = laplacian_nd_inplace(c_nd, _lap_c)
        lap_phi = laplacian_nd_inplace(phi_nd, _lap_phi)
        lap_h = laplacian_nd_inplace(h_phi, _lap_h)

        # time derivatives
        dc_dt = P_CH * (lap_c - lap_h)
        dphi_dt = (P_AC1 * (c_nd - h_phi) * dh_dphi +
                   P_AC2 * 2.0 * phi_nd * (1.0 - phi_nd) * (2.0 * phi_nd - 1.0) +
                   P_AC3 * lap_phi)

        # zero derivatives at boundaries (preserve original behavior)
        dc_dt[0] = dc_dt[-1] = 0.0
        dphi_dt[0] = dphi_dt[-1] = 0.0

        return np.concatenate([dc_dt, dphi_dt])

    # --- Initial condition vector ---
    y0_nd = np.concatenate([c_ic_nd(xs_nd), phi_ic_nd(xs_nd)])

    # --- Solve ODE system ---
    sol = solve_ivp(
        fun=rhs_nd,
        t_span=(ts_nd[0], ts_nd[-1]),
        y0=y0_nd,
        method='BDF',
        t_eval=ts_nd,
        rtol=1e-7,
        atol=1e-9
    )

    y_nd_sol = sol.y.T

    end_time = time.perf_counter()
    solve_duration = end_time - start_time

    return {
      "solution": {
        "phi": y_nd_sol[:, nx:].astype(np.float32),
        "c":   y_nd_sol[:, :nx].astype(np.float32)
      },
      "solve_time": solve_duration
    }



# ==============================================================================
# --- 4. PARAMETER GENERATION
# ==============================================================================

def generate_parameter_sets(param_base: Dict, span_pde: Dict, span_model: Dict, total_simulations: int) -> List[Dict]:
    """
    Generates a list of complete simulation parameter dictionaries using Sobol sampling.
    """
    print(f"Generating {total_simulations} parameter sets...")
    param_names = list(span_pde.keys())
    num_params = len(param_names)

    # 1. Generate points in the normalized [0, 1] space
    sampler = qmc.Sobol(d=num_params, scramble=True)
    model_points = sampler.random(n=total_simulations)

    # 2. Directly create the final list of parameter dictionaries.
    param_list = [
        {
            **param_base,
            **{
                name: map_span(model_points[i, j], span_model[name], span_pde[name])
                for j, name in enumerate(param_names)
            }
        }
        for i in range(total_simulations)
    ]
    
    print("--- Parameter generation complete. ---")
    return param_list


# ==============================================================================
# --- 5. SIMULATION EXECUTION & STORAGE
# ==============================================================================

def run_and_save_simulations(filename: str, param_names_to_store: Set[str], param_list: List[Dict]):
    start_total_time = time.perf_counter()
    total_simulations = len(param_list)

    if not param_list:
        print("Warning: param_list is empty. Nothing to simulate.")
        return

    # Use a try/finally block to ensure the summary always prints
    try:
        with h5py.File(filename, 'w') as hf:
            # All constant parameters can be taken from the first simulation's dict
            param_base = param_list[0]
            nt, nx = param_base['nt'], param_base['nx']

            print(f"\n--- Starting Simulation Run ({total_simulations} total) ---")
            print(f"Results will be saved to '{filename}'")

            print("-> Allocating disk space for simulation results...")
            dset_phi = hf.create_dataset('phi', shape=(total_simulations, nt, nx), dtype='float32', fillvalue=np.nan)
            dset_c = hf.create_dataset('c', shape=(total_simulations, nt, nx), dtype='float32', fillvalue=np.nan)
            dset_times = hf.create_dataset('solve_times', shape=(total_simulations,), dtype='float16', fillvalue=-1.0)

            print(f"-> Storing the {len(param_names_to_store)} specified PDE parameters...")
            for name in sorted(list(param_names_to_store)): # Sorting for consistent order
                try:
                    values = [p[name] for p in param_list]
                    dataset_name = f'{name}'
                    hf.create_dataset(dataset_name, data=values, dtype='float64')
                    print(f"   - Created dataset '{dataset_name}'")
                except KeyError:
                    print(f"   - WARNING: Parameter '{name}' not found in parameter sets. Skipping.")

            print("-> Storing constant axes 'x_nd' and 't_nd'...")
            x_nd = np.linspace(param_base['x_range'][0] / param_base['l_0'], param_base['x_range'][1] / param_base['l_0'], nx)
            t_nd = np.linspace(param_base['t_range'][0] / param_base['t_0'], param_base['t_range'][1] / param_base['t_0'], nt)
            hf.create_dataset('x_nd', data=x_nd, dtype='float64')
            hf.create_dataset('t_nd', data=t_nd, dtype='float64')

            max_workers = os.cpu_count()
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(solve_pde, p): i for i, p in enumerate(param_list)}
                print(f"\nSubmitting {total_simulations} simulations to the process pool ({max_workers} workers)...")

                for future in tqdm(concurrent.futures.as_completed(futures), total=total_simulations, desc="Simulations"):
                    sim_index = futures[future]
                    try:
                        result = future.result()
                        dset_phi[sim_index, :, :] = result['solution']['phi']
                        dset_c[sim_index, :, :] = result['solution']['c']
                        dset_times[sim_index] = result['solve_time']
                    except Exception as e:
                        # --- THIS IS THE ENHANCED ERROR REPORTING BLOCK ---
                        
                        # 1. Retrieve the parameters that caused the failure
                        failed_params = param_list[sim_index]
                        
                        # 2. Print a detailed, multi-line error message
                        print(f"\n--- ERROR: Simulation index {sim_index} FAILED ---")
                        print(f"  Exception Type: {type(e).__name__}")
                        print(f"  Error Message: {e}")
                        print(f"  Failing Parameters: {failed_params}")
                        print("--------------------------------------------------\n")
                        
                        # 3. The HDF5 datasets will automatically be filled with the
                        #    'fillvalue' (np.nan) we specified at creation, so no
                        #    explicit write is needed here for the failed data.

    finally:
        end_total_time = time.perf_counter()
        print("\n----------------------------------------------------")
        print("All simulations complete.")
        print(f"Total execution time: {(end_total_time - start_total_time) / 60:.2f} minutes")
        print("----------------------------------------------------")


def load_h5(filename: str, indices: Optional[Union[int, List[int], Set[int]]] = None) -> Dict[str, Any]:
    """
    Loads all datasets from an HDF5 file into a dictionary of JAX arrays.
    """
    data_dict = {}

    with h5py.File(filename, 'r') as hf:
        # --- 1. Determine which indices to load ---
        if indices is None:
            # Slice(None) is used to select all elements along an axis
            indices_to_load = slice(None)
        elif isinstance(indices, int):
            indices_to_load = indices
        else:
            # Ensure indices are sorted and unique for efficient HDF5 slicing
            indices_to_load = sorted(list(set(indices)))

        # --- 2. Dynamically discover and load all datasets ---
        for key in hf.keys():
            # Clean up the key for the output dictionary (e.g., 'L_pde' -> 'L')
            output_key = key.removesuffix('_pde')
            
            dataset = hf[key]
            
            # Load data based on dataset dimensions
            if dataset.ndim == 1 and key not in ['x_nd', 't_nd']:
                # 1D array of parameters (e.g., 'L_pde', 'solve_times')
                data = dataset[indices_to_load]
            elif dataset.ndim > 1:
                data = dataset[indices_to_load, ...]
            else:
                data = dataset[:]

            data_dict[output_key] = data

    # --- 3. Convert to JAX arrays and stop gradients ---
    for key, value in data_dict.items():
        data_dict[key] = jax.lax.stop_gradient(jnp.asarray(value))
        
    # Rename axes for consistency if they exist
    if 'x_nd' in data_dict:
        data_dict['x'] = data_dict.pop('x_nd')
    if 't_nd' in data_dict:
        data_dict['t'] = data_dict.pop('t_nd')

    return data_dict