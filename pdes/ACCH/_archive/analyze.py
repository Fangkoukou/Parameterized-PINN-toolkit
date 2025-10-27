# """
# HDF5 Simulation Analysis and Visualization Toolkit.

# This module provides a suite of tools for processing and analyzing large datasets
# of PDE simulations stored in HDF5 files. The key functionalities include:

# 1.  **Data Loading**: An efficient HDF5 loader that can selectively load data
#     slices into JAX arrays.
# 2.  **Contour Analysis**: A core `analyze` function that uses `skimage` to find
#     the boundary contour (e.g., the phi=0.5 level set) in a simulation and
#     calculates its geometric properties like arc length and curvature.
# 3.  **Parallel Processing**: A `batch_analyze` function that leverages Python's
#     `multiprocessing` to run the analysis on all simulations in a file in
#     parallel, significantly speeding up the workflow.
# 4.  **Visualization**:
#     - A `demo` function for visual inspection of a single simulation, showing
#       the solution fields and the extracted boundary.
#     - A `make_pde_property_heatmaps` function to create 2D heatmaps of the
#       analyzed properties (e.g., solve time, curvature) across the parameter
#       space (L vs. M).
# """

# # ==============================================================================
# # 1. Imports
# # ==============================================================================

# import multiprocessing
# import os
# from typing import Any, Dict, List, Optional, Union

# import h5py
# import jax
# import jax.numpy as jnp
# import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal import savgol_filter
# from skimage import measure
# from tqdm import tqdm

# # ==============================================================================
# # 2. Data Loading
# # ==============================================================================


# def load_h5(filename, indices = None, data_only = True):
#     """
#     Loads specified datasets from an HDF5 file into a dictionary of JAX arrays.

#     This function can load all entries, a single entry by index, or a list of
#     indices. It automatically handles different dataset dimensions and converts
#     the final data to JAX arrays with gradients stopped.

#     Args:
#         filename (str): The path to the HDF5 file.
#         indices (int or list, optional): The index or indices to load. If None,
#                                           all entries are loaded. Defaults to None.
#         data_only (bool): If True, utility datasets like 'solve_times' are skipped.
#                           Defaults to True.

#     Returns:
#         A dictionary mapping dataset names to their corresponding JAX array data.
#     """
#     data_dict = {}
#     with h5py.File(filename, 'r') as hf:
#         indices_to_load = slice(None) if indices is None else indices

#         for key in hf.keys():
#             if data_only and key == 'solve_times':
#                 continue

#             output_key = key.removesuffix('_pde')
#             dataset = hf[key]

#             if dataset.ndim > 1 or (dataset.ndim == 1 and key not in ['x_nd', 't_nd']):
#                 data = dataset[indices_to_load, ...]
#             else:
#                 data = dataset[:]
#             data_dict[output_key] = data

#     for key, value in data_dict.items():
#         data_dict[key] = jax.lax.stop_gradient(jnp.asarray(value))

#     if 'x_nd' in data_dict: data_dict['x'] = data_dict.pop('x_nd')
#     if 't_nd' in data_dict: data_dict['t'] = data_dict.pop('t_nd')

#     return data_dict


# # ==============================================================================
# # 3. Core Analysis Functions
# # ==============================================================================


# def analyze(filename, index):
#     """
#     Analyzes a single simulation to extract and measure its boundary contour.

#     This function loads one simulation entry, finds the contour of the `phi`
#     field using `skimage.measure.find_contours`, and then calculates geometric
#     properties of this contour, such as arc length and curvature.

#     Args:
#         filename (str): The path to the HDF5 file.
#         index (int): The index of the simulation entry to analyze.

#     Returns:
#         A dictionary containing the analysis results (parameters, coordinates,
#         and geometric properties), or None if an error occurs.
#     """
#     try:
#         data = load_h5(filename, index, data_only=False)
#         x, t, L, M, phi, solve_time = (data[k] for k in ['x', 't', 'L', 'M', 'phi', 'solve_time'])
#     except Exception as e:
#         print(f"Error loading data for index {index}: {e}")
#         return None

#     if phi.min() == phi.max():
#         return {'L': L, 'M': M, 'solve_time': solve_time, 'coordinates': None}

#     contour_level = np.mean([phi.min(), phi.max()])
#     contours = measure.find_contours(phi, level=contour_level)
#     if not contours:
#         return {'L': L, 'M': M, 'solve_time': solve_time, 'coordinates': None}

#     pixel_coords = max(contours, key=len)
#     x_range, y_range = (x.min(), x.max()), (t.min(), t.max())
#     num_rows, num_cols = phi.shape
#     y_coords = y_range[0] + (pixel_coords[:, 0] / (num_rows - 1)) * (y_range[1] - y_range[0])
#     x_coords = x_range[0] + (pixel_coords[:, 1] / (num_cols - 1)) * (x_range[1] - x_range[0])
#     final_coords = np.vstack((x_coords, y_coords)).T

#     window_length = 51
#     if len(x_coords) > window_length:
#         x_smooth = savgol_filter(x_coords, window_length, 3)
#         y_smooth = savgol_filter(y_coords, window_length, 3)
#     else:
#         x_smooth, y_smooth = x_coords, y_coords

#     dx_dt, dy_dt = np.gradient(x_smooth), np.gradient(y_smooth)
#     d2x_dt2, d2y_dt2 = np.gradient(dx_dt), np.gradient(dy_dt)
#     numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
#     denominator = (dx_dt**2 + dy_dt**2)**1.5
#     curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
#     segment_lengths = np.sqrt(np.sum(np.diff(final_coords, axis=0)**2, axis=1))

#     return {
#         'L': L, 'M': M, 'solve_time': solve_time, 'coordinates': final_coords,
#         'mean_curvature': np.mean(curvature), 'max_curvature': np.max(curvature),
#         'arc_length': np.sum(segment_lengths)
#     }


# # ==============================================================================
# # 4. Parallel Batch Processing
# # ==============================================================================


# def worker_wrapper(filename, index):
#     """
#     A robust wrapper for the `analyze` function for use with `multiprocessing`.

#     This function calls `analyze` and formats the output. If `analyze` fails or
#     finds no contour, it loads the basic parameters and returns a dictionary
#     filled with NaNs for the analysis results, ensuring a consistent output
#     structure.

#     Args:
#         filename (str): The path to the HDF5 file.
#         index (int): The index of the simulation entry to analyze.

#     Returns:
#         A dictionary of scalar analysis results, or None on critical failure.
#     """
#     try:
#         results = analyze(filename, index)
#         if results is None or results.get('coordinates') is None:
#             data = load_h5(filename, index, data_only=False)
#             return {'index': index, 'L': data['L'], 'M': data['M'], 'solve_time': np.nan,
#                     'mean_curvature': np.nan, 'max_curvature': np.nan, 'arc_length': np.nan}
#         return {
#             'index': index, 'L': results['L'], 'M': results['M'],
#             'solve_time': results['solve_time'], 'mean_curvature': results['mean_curvature'],
#             'max_curvature': results['max_curvature'], 'arc_length': results['arc_length']
#         }
#     except Exception as e:
#         print(f"Critical error in worker for index {index}: {e}")
#         return None


# def batch_analyze(filename):
#     """
#     Processes all simulation entries in an HDF5 file in parallel.

#     This function determines the number of entries in the file, sets up a
#     multiprocessing pool using all available CPU cores, and runs the analysis
#     on each entry. It displays a progress bar using `tqdm`.

#     Args:
#         filename (str): The path to the HDF5 file to process.

#     Returns:
#         A dictionary where each key corresponds to an analysis metric (e.g.,
#         'arc_length') and the value is a NumPy array of that metric for all
#         processed entries. Returns None if the file cannot be read or no
#         entries are successfully processed.
#     """
#     try:
#         with h5py.File(filename, 'r') as hf:
#             num_entries = len(hf['L'])
#     except (IOError, KeyError) as e:
#         print(f"Error reading input file {filename}: {e}")
#         return None

#     print(f"Found {num_entries} entries to process in '{filename}'.")
#     tasks = [(filename, i) for i in range(num_entries)]
#     num_cores = multiprocessing.cpu_count()
#     print(f"Starting analysis on {num_cores} cores...")

#     with multiprocessing.Pool(processes=num_cores) as pool:
#         results_iterator = pool.starmap(worker_wrapper, tasks)
#         all_results = [res for res in tqdm(results_iterator, total=num_entries, desc="Analyzing entries") if res is not None]

#     if not all_results:
#         print("No entries were successfully processed.")
#         return None

#     print(f"\nAnalysis complete. Processed {len(all_results)} of {num_entries} entries.")
#     return {key: np.array([res[key] for res in all_results]) for key in all_results[0]}


# # ==============================================================================
# # 5. Visualization
# # ==============================================================================


# def demo(filename, index):
#     """
#     Performs analysis and provides a detailed visual check for a single entry.

#     This function runs the `analyze` function on a single simulation index,
#     prints a summary of the geometric properties, and then generates a
#     side-by-side plot of the `phi` and `c` fields with the extracted boundary
#     contour overlaid for easy verification.

#     Args:
#         filename (str): The path to the HDF5 file.
#         index (int): The index of the simulation entry to demonstrate.
#     """
#     results = analyze(filename, index)
#     if results is None or results.get('coordinates') is None:
#         print(f"\nAnalysis failed for index {index}. Cannot generate plot.")
#         return

#     print("\n--- Analysis Summary ---")
#     print(f"Parameters: L = {results['L']:.2f}, M = {results['M']:.2f}")
#     print(f"Solve Time: {results['solve_time']:.4f} s")
#     print(f"Arc Length: {results['arc_length']:.4f}, Mean Curvature: {results['mean_curvature']:.4f}")

#     print("\nGenerating side-by-side plots for visual verification...")
#     data = load_h5(filename, index, data_only=False)
#     x, t, c, phi = data['x'], data['t'], data['c'], data['phi']
#     plot_extent = [x.min(), x.max(), t.min(), t.max()]
#     boundary_coords = results['coordinates']

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
#     fig.suptitle(f'Visual Check for {os.path.basename(filename)} (Index: {index})', fontsize=16)

#     im1 = ax1.imshow(phi, cmap='viridis', origin='lower', extent=plot_extent, aspect='auto')
#     ax1.plot(boundary_coords[:, 0], boundary_coords[:, 1], 'r-', lw=2.5, label='Extracted Boundary')
#     fig.colorbar(im1, ax=ax1, label='Phi Value')
#     ax1.set_title('Phi Field with Boundary'); ax1.set_xlabel('X-axis'); ax1.set_ylabel('T-axis'); ax1.legend()

#     im2 = ax2.imshow(c, cmap='viridis', origin='lower', extent=plot_extent, aspect='auto')
#     ax2.plot(boundary_coords[:, 0], boundary_coords[:, 1], 'r-', lw=2.5, label='Extracted Boundary')
#     fig.colorbar(im2, ax=ax2, label='C Value')
#     ax2.set_title('C Field with Boundary'); ax2.set_xlabel('X-axis'); ax2.legend()

#     plt.tight_layout()
#     plt.show()


# def make_pde_property_heatmaps(analysis_results, out_dir, plotname, use_log = False, nlevels = 100):
#     """
#     Generates and saves heatmaps of analyzed properties across the L-M parameter space.

#     For each specified property (e.g., 'solve_time', 'arc_length'), this function
#     creates a 2D contour plot showing how that property varies as a function of
#     the 'L' and 'M' simulation parameters.

#     Args:
#         analysis_results (dict): The dictionary returned by `batch_analyze`.
#         out_dir (str): The directory where the output plots will be saved.
#         plotname (str): A base name for the output plot files (e.g., 'run1_analysis').
#         use_log (bool): If True, all plots will use a logarithmic color scale.
#         nlevels (int): The number of contour levels for the heatmap.
#     """
#     print(f"\nGenerating property heatmaps (Log Scale: {use_log})...")
#     os.makedirs(out_dir, exist_ok=True)
#     plt.style.use('seaborn-v0_8-poster')

#     properties_to_plot = {
#         'solve_time': 'Solve Time (s)', 'mean_curvature': 'Mean Curvature',
#         'max_curvature': 'Maximum Curvature', 'arc_length': 'Arc Length'
#     }

#     for prop_key, prop_title in properties_to_plot.items():
#         print(f"  - Plotting {prop_title}...")
#         valid_mask = ~np.isnan(analysis_results[prop_key])
#         L, M, values = (analysis_results[k][valid_mask] for k in ['L_pde', 'M_pde', prop_key])

#         if len(values) == 0:
#             print(f"    Skipping {prop_title} due to no valid data."); continue

#         vmin, vmax = values.min(), values.max()
#         if vmin == vmax: vmin, vmax = vmin - 0.1, vmax + 0.1

#         title, scale_suffix = f'Heatmap of {prop_title}', 'linear'
#         if use_log and vmin > 0:
#             norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
#             levels = np.logspace(np.log10(vmin), np.log10(vmax), nlevels)
#             title += " (Log Scale)"; scale_suffix = 'log'
#         else:
#             if use_log: print(f"    Warning: Cannot use log scale for '{prop_title}'. Falling back to linear.")
#             norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
#             levels = np.linspace(vmin, vmax, nlevels)

#         fig, ax = plt.subplots(figsize=(20, 15))
#         try:
#             im = ax.tricontourf(L, M, values, levels=levels, cmap='inferno', norm=norm)
#         except Exception as e:
#             print(f"    tricontourf failed ({e}), falling back to scatter plot.")
#             im = ax.scatter(L, M, c=values, cmap='inferno', norm=norm, s=20)

#         ax.set_title(title, fontsize=28, pad=25)
#         ax.set_xlabel('L Parameter', fontsize=22); ax.set_ylabel('M Parameter', fontsize=22)
#         fig.colorbar(im, ax=ax).set_label(prop_title, fontsize=22)

#         output_path = os.path.join(out_dir, f'{plotname}_{prop_key}_{scale_suffix}.png')
#         plt.savefig(output_path, dpi=120, bbox_inches='tight')
#         print(f"    Saved plot to '{output_path}'")
#         plt.close(fig)

#     print("\nAll heatmaps generated successfully.")