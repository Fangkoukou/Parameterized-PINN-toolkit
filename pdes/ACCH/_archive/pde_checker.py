# """
# PDE Simulation Visualization and Verification Toolkit.

# This module provides the `PDE_checker` class, a utility for visualizing,
# comparing, and quantitatively verifying the results of PDE simulations. It is
# designed to work with solution dictionaries containing 'x', 't', 'phi', and 'c'
# arrays.

# The main functionalities include:
# - Standard visualizations: 2D heatmaps and 1D temporal snapshots.
# - Comparative analysis: Visual overlays and quantitative error diagnostics
#   to compare two different simulation results.
# """

# # ==============================================================================
# # 1. Imports
# # ==============================================================================

# import jax.numpy as jnp
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import LogNorm
# from matplotlib.lines import Line2D

# # ==============================================================================
# # 2. PDE Checker Class
# # ==============================================================================


# class PDE_checker:
#     """
#     A utility class for visualizing and verifying PDE simulation results.

#     This class offers several methods to plot and compare simulation outputs,
#     making it easier to debug solvers, compare different models, or simply
#     inspect the dynamics of the phase field (`phi`) and concentration (`c`).
#     """

#     # ==========================================================================
#     # 3. Standard Visualization Methods
#     # ==========================================================================

#     def heatmaps(self, sol, title=""):
#         """
#         Generates 2D heatmaps for the phi and c solution fields over space and time.

#         Args:
#             sol (dict): A solution dictionary containing 'x', 't', 'c', and 'phi' arrays.
#             title (str): An optional prefix for the plot titles.
#         """
#         fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
#         fig.suptitle(f'PDE Simulation Results: {title}', fontsize=14)

#         # Extract simulation data and define plot boundaries
#         x, t, c, phi = sol["x"], sol["t"], sol["c"], sol["phi"]
#         extent = [x[0], x[-1], t[0], t[-1]]

#         # --- Phi Heatmap ---
#         im1 = ax[0].imshow(phi, aspect='auto', origin='lower', extent=extent)
#         ax[0].set_title(f"{title} phi")
#         ax[0].set_xlabel("Position (x)")
#         ax[0].set_ylabel("Time (t)")
#         fig.colorbar(im1, ax=ax[0], label="phi")

#         # --- C Heatmap ---
#         im2 = ax[1].imshow(c, aspect='auto', origin='lower', extent=extent)
#         ax[1].set_title(f"{title} c")
#         ax[1].set_xlabel("Position (x)")
#         ax[1].set_ylabel("Time (t)")
#         fig.colorbar(im2, ax=ax[1], label="c")

#         plt.show()

#     def frames(self, sol, title="", num_frames=5):
#         """
#         Plots 1D snapshots (frames) of phi and c at selected time points.

#         This is useful for seeing the evolution of the 1D profiles over time.

#         Args:
#             sol (dict): A solution dictionary containing 'x', 't', 'c', and 'phi' arrays.
#             title (str): An optional prefix for the plot titles.
#             num_frames (int): The number of time snapshots to display.
#         """
#         fig, (ax_phi, ax_c) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
#         fig.suptitle(f'PDE Simulation Frames: {title}', fontsize=14)

#         x, t, c, phi = sol["x"], sol["t"], sol["c"], sol["phi"]

#         # Select evenly spaced time indices and corresponding colors
#         time_indices = jnp.linspace(0, len(t) - 1, num_frames, dtype=int)
#         colors = cm.viridis(np.linspace(0, 1, num_frames))

#         # Use a spec to reduce code duplication for plotting phi and c
#         plot_specs = [
#             {'ax': ax_phi, 'data': phi, 'title_suffix': 'phi', 'ylabel': 'phi'},
#             {'ax': ax_c,   'data': c,   'title_suffix': 'c',   'ylabel': 'c'}
#         ]

#         for spec in plot_specs:
#             ax = spec['ax']
#             for i, time_idx in enumerate(time_indices):
#                 ax.plot(x, spec['data'][time_idx, :],
#                         color=colors[i], linewidth=2.5,
#                         label=f't = {t[time_idx]:.2e}')
#             ax.set_title(f"{title} {spec['title_suffix']}")
#             ax.set_xlabel("Position (x)")
#             ax.set_ylabel(spec['ylabel'])
#             ax.grid(True, linestyle=':', alpha=0.7)
#             ax.legend(loc='best', fontsize=9, title="Time")

#         plt.show()

#     # ==========================================================================
#     # 4. Comparative Analysis Methods
#     # ==========================================================================

#     def check1(self, results1, results2, label1="Solver 1", label2="Solver 2", num_frames=5):
#         """
#         Visually compares two solver results with a side-by-side overlay plot.

#         This method plots the profiles from the first solver as thick solid lines
#         and overlays the profiles from the second solver as thin dashed lines,
#         making it easy to spot any visual discrepancies.

#         Args:
#             results1 (dict): Solution dictionary from the first solver.
#             results2 (dict): Solution dictionary from the second solver.
#             label1 (str): A descriptive label for the first solver.
#             label2 (str): A descriptive label for the second solver.
#             num_frames (int): The number of time snapshots to compare.
#         """
#         x1, t1, c1, phi1 = results1["x"], results1["t"], results1["c"], results1["phi"]
#         x2, c2, phi2 = results2["x"], results2["c"], results2["phi"]

#         time_indices = jnp.linspace(0, len(t1) - 1, num_frames, dtype=int)
#         colors = cm.viridis(np.linspace(0, 1, num_frames))

#         fig, (ax_phi, ax_c) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
#         fig.suptitle('Comparison of Solution Profiles Over Time', fontsize=14)

#         plot_specs = [
#             {'ax': ax_phi, 'data1': phi1, 'data2': phi2, 'title': r'Phase Field, $\phi$'},
#             {'ax': ax_c,   'data1': c1,   'data2': c2,   'title': 'Concentration, c'}
#         ]

#         for spec in plot_specs:
#             ax = spec['ax']
#             for i, time_idx in enumerate(time_indices):
#                 time_label = f't = {t1[time_idx]:.2e}'
#                 # Plot primary solution (thick, solid)
#                 ax.plot(x1, spec['data1'][time_idx, :], color=colors[i],
#                         linestyle='-', linewidth=4, alpha=0.9, label=time_label)
#                 # Plot comparison solution (thin, dashed overlay)
#                 ax.plot(x2, spec['data2'][time_idx, :], color='red',
#                         linestyle='--', linewidth=2, alpha=0.65)
#             ax.set_title(spec['title'])
#             ax.set_xlabel('Position (x)')
#             ax.grid(True, linestyle=':', alpha=0.7)

#         ax_phi.set_ylabel(r'$\phi$')
#         ax_c.set_ylabel('Concentration')

#         # --- Build a clear, combined legend ---
#         handles, labels = ax_phi.get_legend_handles_labels()
#         overlay_handle = Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'{label2} (Overlay)')
#         handles.append(overlay_handle)
#         ax_phi.legend(handles=handles, loc='best', title=label1)

#         plt.tight_layout(rect=[0, 0, 1, 0.94])
#         plt.show()

#     def _plot_1d_diagnostic(self, x_coords, sol1, sol2, abs_diff, t_fail,
#                             var_name, var_symbol, label1, label2, log_scale):
#         """
#         Internal helper to plot a 1D comparison and its absolute error at a single time.
#         """
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
#                                        gridspec_kw={'height_ratios': [3, 1]})
#         fig.suptitle(f'Discrepancy in {var_name} ({var_symbol}) at t = {t_fail:.2e}', fontsize=16)

#         # Top panel: Solution overlay
#         ax1.plot(x_coords, sol1, 'b-', lw=4, alpha=0.7, label=label1)
#         ax1.plot(x_coords, sol2, 'r--', lw=2, label=label2)
#         ax1.set_ylabel(var_symbol)
#         ax1.legend()
#         ax1.grid(True, linestyle=':')

#         # Bottom panel: Absolute error
#         ax2.plot(x_coords, abs_diff, 'k-')
#         ax2.set_xlabel('Position (x)')
#         ax2.set_ylabel(f'Absolute Error |Δ{var_symbol}|')
#         ax2.grid(True, linestyle=':')
#         if log_scale:
#             ax2.set_yscale('log')

#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()

#     def check2(self, results1, results2, label1="Reference", label2="Test",
#                log_scale=False, rtol=1e-6, atol=1e-8):
#         """
#         Performs a comprehensive quantitative verification between two solver results.

#         This method calculates the absolute difference between two solutions,
#         checks if they are close within a given tolerance, and generates detailed
#         diagnostic plots, including 2D error heatmaps and 1D error profiles at
#         the point of maximum discrepancy.

#         Args:
#             results1 (dict): The reference solution dictionary.
#             results2 (dict): The solution dictionary to compare against the reference.
#             label1 (str): A label for the reference solution.
#             label2 (str): A label for the test solution.
#             log_scale (bool): If True, use a logarithmic scale for error plots.
#             rtol (float): The relative tolerance for `jnp.allclose`.
#             atol (float): The absolute tolerance for `jnp.allclose`.
#         """
#         print("--- Starting Comprehensive Solver Verification ---")
#         all_passed = True
#         x_coords, t_coords = results1["x"], results1["t"]
#         abs_diff_phi = jnp.abs(results1["phi"] - results2["phi"])
#         abs_diff_c = jnp.abs(results1["c"] - results2["c"])

#         variables_to_check = [
#             {"name": "Phase Field", "symbol": "φ", "diff": abs_diff_phi, **results1, **results2},
#             {"name": "Concentration", "symbol": "c", "diff": abs_diff_c, **results1, **results2},
#         ]

#         for i, var in enumerate(variables_to_check):
#             print(f"\n[{i+1}] Verifying {var['name']} solution '{var['symbol']}'...")
#             max_abs_diff = jnp.max(var["diff"])
#             t_idx, x_idx = jnp.unravel_index(jnp.argmax(var["diff"]), var["diff"].shape)
#             print(f"   - Max absolute difference |Δ{var['symbol']}|: {max_abs_diff:.2e}")
#             print(f"   - Location: time_index={t_idx}, space_index={x_idx}")

#             # --- Tolerance Check ---
#             if jnp.allclose(var["data1"], var["data2"], rtol=rtol, atol=atol):
#                 print("   - SUCCESS: Solutions are within tolerance.")
#             else:
#                 all_passed = False
#                 print("   - FAILURE: Solutions exceed tolerance.")

#             # --- 1D Diagnostic Plot ---
#             t_fail = t_coords[t_idx]
#             print(f"   - Generating 1D diagnostic plot at t = {t_fail:.2e}...")
#             self._plot_1d_diagnostic(
#                 x_coords, var["data1"][t_idx, :], var["data2"][t_idx, :],
#                 var["diff"][t_idx, :], t_fail,
#                 var["name"], var["symbol"], label1, label2, log_scale
#             )

#         # --- 2D Error Heatmaps ---
#         print("\n[3] Generating 2D absolute error heatmaps...")
#         fig_hm, (ax_phi_hm, ax_c_hm) = plt.subplots(1, 2, figsize=(14, 6))
#         fig_hm.suptitle(f'Absolute Error Heatmaps |{label1} - {label2}|', fontsize=16)
#         extent = [x_coords[0], x_coords[-1], t_coords[0], t_coords[-1]]

#         # Use LogNorm for better visualization of small errors
#         norm_phi = LogNorm() if log_scale and jnp.any(abs_diff_phi > 0) else None
#         norm_c = LogNorm() if log_scale and jnp.any(abs_diff_c > 0) else None

#         im1 = ax_phi_hm.imshow(abs_diff_phi, aspect='auto', origin='lower', extent=extent, cmap='magma', norm=norm_phi)
#         ax_phi_hm.set_title('Absolute Error in φ')
#         ax_phi_hm.set_xlabel('Position (x)'); ax_phi_hm.set_ylabel('Time')
#         fig_hm.colorbar(im1, ax=ax_phi_hm, label='|Δφ|')

#         im2 = ax_c_hm.imshow(abs_diff_c, aspect='auto', origin='lower', extent=extent, cmap='magma', norm=norm_c)
#         ax_c_hm.set_title('Absolute Error in c')
#         ax_c_hm.set_xlabel('Position (x)'); ax_c_hm.set_ylabel('Time')
#         fig_hm.colorbar(im2, ax=ax_c_hm, label='|Δc|')

#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()

#         # --- Final Summary ---
#         print("\n--- Verification Complete ---")
#         if all_passed:
#             print("✅ Final Result: All checks passed within tolerance.")
#         else:
#             print("❌ Final Result: Some checks failed. See diagnostic plots.")