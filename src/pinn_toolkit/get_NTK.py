"""
Neural Tangent Kernel (NTK) Computation and Visualization Module.

This module provides a powerful and memory-efficient function `get_NTK` for
computing and visualizing the block-wise NTK matrix of a PINN. It is designed
to handle very large datasets that may not fit into GPU or CPU RAM by offering
several strategies:

1.  **In-Memory Caching**: Jacobians for smaller residual components are computed
    once and cached in memory.
2.  **Streaming Computation**: For components whose Jacobians exceed a RAM limit,
    the NTK blocks are computed in a streaming fashion, batch by batch.
3.  **Downsampled Visualization**: When streaming, the NTK block is visualized
    as a downsampled image to avoid creating a massive matrix in memory.
4.  **Flexible Plotting**: Users can choose to plot the entire block-wise NTK
    in a single grid or save each block as a separate image file to conserve
    memory during plotting.
"""

# ==============================================================================
# 1. Imports
# ==============================================================================

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, lax, vmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_map
from matplotlib.colors import SymLogNorm

# ==============================================================================
# 2. Helper Functions
# ==============================================================================


@jit
def _get_jacobian_batched(res_fn_wrapped, inp_dict, flat_p, recon, static, batch_size):
    """
    Computes the Jacobian of a residual function w.r.t. flattened model
    parameters in a memory-efficient, batched manner.

    This function is JIT-compiled for performance.

    Args:
        res_fn_wrapped: The NTK-compatible residual function.
        inp_dict (dict): The PyTree of input data for the residual function.
        flat_p (jnp.ndarray): The flattened model parameters.
        recon: The reconstruction function from `ravel_pytree`.
        static: The static part of the Equinox model.
        batch_size (int): The number of points to process in each batch.

    Returns:
        jnp.ndarray: The computed Jacobian matrix of shape (n_residuals, n_params).
    """
    input_leaves = tree_leaves(inp_dict)
    if not input_leaves or input_leaves[0].shape[0] == 0:
        return jnp.empty((0, flat_p.shape[0]), dtype=flat_p.dtype)

    n_points = input_leaves[0].shape[0]

    # Define the function to compute the Jacobian for a single data point
    def compute_jac_single(inp_slice):
        local_fn = lambda p: res_fn_wrapped(p, recon, static, inp_slice)
        return jax.jacrev(local_fn)(flat_p)

    # Vectorize the single-point Jacobian computation
    compute_jac_batch = vmap(compute_jac_single, in_axes=(tree_map(lambda _: 0, inp_dict),))

    # --- Batching logic using lax.scan for JIT-compatibility ---
    num_batches = (n_points + batch_size - 1) // batch_size
    
    def scan_body(carry, i):
        start_idx = i * batch_size
        # Use dynamic_slice for variable batch sizes (especially the last one)
        inp_batch = tree_map(
            lambda x: lax.dynamic_slice_in_dim(x, start_idx, min(batch_size, n_points - start_idx)),
            inp_dict
        )
        return carry, compute_jac_batch(inp_batch)

    _, J_batches = lax.scan(scan_body, None, jnp.arange(num_batches))
    
    # The output of scan will have a leading dimension for batches
    J_full = jnp.concatenate(J_batches, axis=0)

    # Reshape from (n_points, n_outputs_per_point, n_params) to (total_residuals, n_params)
    return J_full.reshape(-1, flat_p.shape[0])


# ==============================================================================
# 3. Main NTK Function
# ==============================================================================


def get_NTK(model, inputs, residual, P_model,
            batch_size=128, use_symlog=False,
            plot_subgrids="line", out_dir=None, plot_name=None, dpi=300,
            selected_blocks=None, ram_limit_gb=4.0, visualization_resolution=(2048, 2048)):
    """
    Computes and visualizes the block-wise Neural Tangent Kernel (NTK) matrix.

    This function provides a robust and memory-aware pipeline for analyzing the NTK,
    which is crucial for understanding the training dynamics of PINNs.

    Args:
        model (eqx.Module): The Equinox model.
        inputs (dict): A dictionary of input points for all residual types.
        residual (Residual): An instance of the Residual class.
        P_model (pytree): A pytree of model parameters (e.g., L, M).
        batch_size (int): The batch size for Jacobian computation.
        use_symlog (bool): If True, uses a symmetric log scale for the color map.
        plot_subgrids (str/bool): Controls sub-grid visualization.
            - "line": Plots one large grid with lines separating parameter sets.
            - "plot": Saves each NTK block as a separate image file (most memory-efficient).
            - False: Plots one large grid with no sub-grid lines.
        out_dir (str, optional): Directory to save plots.
        plot_name (str, optional): Filename for the saved plot.
        dpi (int): Resolution for saved plots.
        selected_blocks (list, optional): A list of blocks to compute (e.g., ['ic', 'bc']).
        ram_limit_gb (float): RAM threshold to switch from caching to streaming computation.
        visualization_resolution (tuple): The (height, width) for downsampled plots.

    Returns:
        dict: A dictionary of the Jacobian matrices that were small enough to be
              cached in memory.
    """
    # --- 1. Initialization and Setup ---
    params, static = eqx.partition(model, eqx.is_inexact_array)
    flat_p, recon = ravel_pytree(params)
    res_fns = residual.ntk_residual_wrappers()

    all_keys = ['ic', 'bc', 'ac', 'ch', 'data']
    blocks_to_process = selected_blocks or all_keys
    inp_key_map = {'ic': 'ic', 'bc': 'bc', 'ac': 'colloc', 'ch': 'colloc', 'data': 'data'}
    output_dims = {'ic': 2, 'bc': 2, 'ac': 1, 'ch': 1, 'data': 2}

    # --- 2. Pre-compute and Cache Jacobians that fit in RAM ---
    jacobians_in_memory = {}
    print("Analyzing memory requirements and caching small Jacobians...")
    for key in blocks_to_process:
        n_points = tree_leaves(inputs[inp_key_map[key]])[0].shape[0]
        # Estimate memory usage in GB (float32 = 4 bytes)
        mem_gb = (n_points * output_dims[key] * flat_p.shape[0] * 4) / (1024**3)
        if mem_gb < ram_limit_gb:
            print(f"  - Computing and caching Jacobian for '{key}' (Est. {mem_gb:.2f} GB)...")
            jacobians_in_memory[key] = _get_jacobian_batched(
                res_fns[key], inputs[inp_key_map[key]], flat_p, recon, static, batch_size
            )
        else:
            print(f"  - Jacobian for '{key}' (Est. {mem_gb:.2f} GB) exceeds RAM limit, will be streamed.")

    # --- 3. Main Logic: Choose Plotting Strategy ---
    if plot_subgrids == "plot":
        # --- STRATEGY A: PLOT INDIVIDUAL FILES (MAXIMUM MEMORY SAVING) ---
        print("Strategy: Generating a separate plot for each NTK block...")
        if not (out_dir and plot_name):
            raise ValueError("`out_dir` and `plot_name` are required when `plot_subgrids='plot'`.")
        os.makedirs(out_dir, exist_ok=True)
        base_name, ext = os.path.splitext(plot_name)

        for key_i in blocks_to_process:
            for key_j in blocks_to_process:
                print(f"  - Processing and plotting block ({key_i}, {key_j})...")
                fig, ax = plt.subplots(figsize=(8, 7))
                block_ntk_cpu = _compute_or_visualize_block(
                    key_i, key_j, jacobians_in_memory, res_fns, inputs, inp_key_map,
                    output_dims, flat_p, recon, static, batch_size, visualization_resolution
                )
                if block_ntk_cpu is None:
                    plt.close(fig); continue
                
                _plot_single_block(ax, block_ntk_cpu, f"NTK Block: {key_i} vs {key_j}", use_symlog)
                
                filepath = os.path.join(out_dir, f"{base_name}_{key_i}-{key_j}{ext}")
                fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"    Saved plot to {filepath}")
    else:
        # --- STRATEGY B: PLOT ONE LARGE GRID ---
        print("Strategy: Generating a single plot with all NTK blocks...")
        num_blocks = len(blocks_to_process)
        fig, axes = plt.subplots(num_blocks, num_blocks, figsize=(4 * num_blocks + 4, 4 * num_blocks + 2),
                                 sharex='col', sharey='row', squeeze=False)
        fig.suptitle("Block-wise NTK Matrix", fontsize=24)

        for i, key_i in enumerate(blocks_to_process):
            axes[i, 0].set_ylabel(key_i, fontsize=18, rotation=90, va='center', labelpad=25)
            for j, key_j in enumerate(blocks_to_process):
                axes[0, j].set_title(key_j, fontsize=18, pad=25)
                ax = axes[i, j]
                block_ntk_cpu = _compute_or_visualize_block(
                    key_i, key_j, jacobians_in_memory, res_fns, inputs, inp_key_map,
                    output_dims, flat_p, recon, static, batch_size, visualization_resolution
                )
                if block_ntk_cpu is None:
                    ax.set_visible(False); continue
                
                _plot_single_block(ax, block_ntk_cpu, "", use_symlog)
                ax.set_xticks([]); ax.set_yticks([])

        fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        if out_dir and plot_name:
            filepath = os.path.join(out_dir, plot_name)
            print(f"Saving plot to {filepath}...")
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    return jacobians_in_memory

# ==============================================================================
# 4. Internal Plotting & Computation Helpers for get_NTK
# ==============================================================================

def _compute_or_visualize_block(key_i, key_j, jacobians_in_memory, res_fns, inputs, inp_key_map,
                                output_dims, flat_p, recon, static, batch_size, viz_res):
    """Internal helper to compute a single NTK block, using streaming if necessary."""
    if key_i in jacobians_in_memory and key_j in jacobians_in_memory:
        # Case 1: Both Jacobians are cached in memory.
        J_i, J_j = jacobians_in_memory[key_i], jacobians_in_memory[key_j]
        if J_i.shape[0] == 0 or J_j.shape[0] == 0: return None
        return np.asarray(J_i @ J_j.T)
    else:
        # Case 2: At least one Jacobian is too large. Stream and create a visualization.
        res_fn_i, inp_i = res_fns[key_i], inputs[inp_key_map[key_i]]
        res_fn_j, inp_j = res_fns[key_j], inputs[inp_key_map[key_j]]
        n_i, n_j = tree_leaves(inp_i)[0].shape[0], tree_leaves(inp_j)[0].shape[0]
        if n_i == 0 or n_j == 0: return None

        image_matrix = np.zeros(viz_res, dtype=np.float64)
        counts_matrix = np.zeros(viz_res, dtype=np.int32)
        
        num_batches_i = (n_i + batch_size - 1) // batch_size
        num_batches_j = (n_j + batch_size - 1) // batch_size

        for b_i in range(num_batches_i):
            start_i, end_i = b_i * batch_size, min((b_i + 1) * batch_size, n_i)
            inp_i_batch = tree_map(lambda x: lax.dynamic_slice_in_dim(x, start_i, end_i - start_i), inp_i)
            J_i_batch = _get_jacobian_batched(res_fn_i, inp_i_batch, flat_p, recon, static, batch_size)
            
            for b_j in range(num_batches_j):
                start_j, end_j = b_j * batch_size, min((b_j + 1) * batch_size, n_j)
                inp_j_batch = tree_map(lambda x: lax.dynamic_slice_in_dim(x, start_j, end_j - start_j), inp_j)
                J_j_batch = _get_jacobian_batched(res_fn_j, inp_j_batch, flat_p, recon, static, batch_size)
                
                # Compute the sub-block NTK and get its average value
                sub_ntk_cpu = np.asarray(J_i_batch @ J_j_batch.T)
                avg_val = np.mean(sub_ntk_cpu)
                
                # Map the sub-block's position to pixel coordinates in the visualization
                start_pr = int((start_i * output_dims[key_i] / (n_i * output_dims[key_i])) * viz_res[0])
                end_pr = int((end_i * output_dims[key_i] / (n_i * output_dims[key_i])) * viz_res[0])
                start_pc = int((start_j * output_dims[key_j] / (n_j * output_dims[key_j])) * viz_res[1])
                end_pc = int((end_j * output_dims[key_j] / (n_j * output_dims[key_j])) * viz_res[1])
                
                # Update the image and count matrices
                image_matrix[start_pr:end_pr, start_pc:end_pc] += avg_val
                counts_matrix[start_pr:end_pr, start_pc:end_pc] += 1
        
        return image_matrix / (counts_matrix + 1e-9) # Avoid division by zero

def _plot_single_block(ax, block_data, title, use_symlog):
    """Internal helper to plot a single NTK block onto a given matplotlib axis."""
    norm = None
    if use_symlog:
        median_abs = np.median(np.abs(block_data))
        linthresh = median_abs / 100 if median_abs > 0 else 1e-5
        norm = SymLogNorm(linthresh=linthresh, vmin=block_data.min(), vmax=block_data.max())
    
    im = ax.imshow(block_data, cmap='viridis', interpolation='nearest', aspect='auto', norm=norm)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title, fontsize=16)