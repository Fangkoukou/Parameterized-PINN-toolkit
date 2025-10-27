"""
PINN Model Evaluation and Visualization Suite.

This script provides a comprehensive toolkit for analyzing the performance of
trained Physics-Informed Neural Network (PINN) models. Its main functionalities
are organized into three parts:

1.  **Model Discovery**: Scans a structured directory to find trained model files
    and their corresponding loss histories based on specified criteria.
2.  **Performance Evaluation**: Loads models and reference data to compute key
    error metrics (L1, L2, Linf) in a memory-efficient, JIT-compiled manner.
3.  **Results Visualization**: Generates a variety of plots to visualize and
    compare model performance, including:
    - Dot plots of average error metrics across different model families.
    - Individual loss history plots for each training run.
    - Aggregated loss history plots that show the average trend for a model family.
"""

# ==============================================================================
# 1. Imports
# ==============================================================================

# --- Standard Python Libraries ---
import gc
import os
import re
import time
import warnings
import h5py
from collections import defaultdict

# --- Core Scientific & Data Libraries ---
import numpy as np

# --- JAX and Machine Learning Ecosystem ---
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, vmap

# --- Visualization Libraries ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Local Project Modules ---
from .util import load_h5, load_model, load_pytree

# ==============================================================================
# 2. Model Discovery
# ==============================================================================


def find_model_paths(models_dir, names=None, methods=None, sizes=None):
    """
    Finds model and loss file paths based on a structured directory layout.

    This function traverses a directory tree expecting the following structure:
    `models_dir/{name}_{method}_{size}/{param_seed}_{train_seed}/`
    where the final directory contains the model and loss files.

    Args:
        models_dir (str): The root directory where models are stored.
        names (list, optional): A list of model base names to filter by.
        methods (list, optional): A list of training methods to filter by.
        sizes (list, optional): A list of data sizes to filter by.

    Returns:
        list: A list of dictionaries, where each dictionary contains metadata
              and file paths for a single discovered model run.

    Raises:
        ValueError: If `methods` is not provided.
    """
    if not os.path.isdir(models_dir):
        warnings.warn(f"Models directory '{models_dir}' not found.")
        return []
    if not methods:
        raise ValueError("A list of 'methods' must be provided.")

    all_models = []
    # Sort methods by length to prevent partial matches (e.g., 'lh' vs 'lhs')
    sorted_methods = sorted(methods, key=len, reverse=True)

    for top_level_dir in os.listdir(models_dir):
        top_level_path = os.path.join(models_dir, top_level_dir)
        if not os.path.isdir(top_level_path):
            continue

        # --- Parse top-level directory name: {name}_{method}_{size} ---
        try:
            base_name, size_str = top_level_dir.rsplit('_', 1)
            size = int(size_str)
        except ValueError:
            continue  # Skip directories that don't match the pattern

        found_method, name = None, None
        for method in sorted_methods:
            if base_name.endswith(f"_{method}"):
                found_method = method
                name = base_name.removesuffix(f"_{method}")
                break

        if not found_method:
            continue
        if (names and name not in names) or (sizes and size not in sizes):
            continue

        # --- Parse seed-level directory name: {param_seed}_{train_seed} ---
        for seed_dir in os.listdir(top_level_path):
            seed_path = os.path.join(top_level_path, seed_dir)
            if not os.path.isdir(seed_path):
                continue

            try:
                param_seed, train_seed = seed_dir.split('_')
            except ValueError:
                continue

            # --- Construct file paths and metadata dictionary ---
            base_filename = f"{name}_{found_method}_{size}_{param_seed}_{train_seed}"
            model_path = os.path.join(seed_path, f"model_{base_filename}.pkl")
            loss_path = os.path.join(seed_path, f"loss_{base_filename}.npz")

            result = {
                'name': name, 'method': found_method, 'size': size,
                'param_seed': param_seed, 'train_seed': train_seed,
                'model_path': model_path if os.path.isfile(model_path) else None,
                'loss_path': loss_path if os.path.isfile(loss_path) else None,
            }

            if result['model_path'] or result['loss_path']:
                all_models.append(result)

    return all_models


# ==============================================================================
# 3. Core Model Evaluation
# ==============================================================================


def compute_errs(refs, preds):
    """Calculates L1, L2, and Linf error metrics for a batch of predictions."""
    diff = preds - refs
    l1_err = jnp.mean(jnp.abs(diff))
    l2_err = jnp.sqrt(jnp.mean(diff**2))
    linf_err = jnp.max(jnp.abs(diff))
    return l1_err, l2_err, linf_err


@eqx.filter_jit
def evaluate_model_performance_single(params, static, ref_data, chunk_size):
    """
    Evaluates a single model's performance against reference data (JIT-compiled).

    This function uses `lax.scan` to iterate over chunks of the reference dataset,
    making it memory-efficient for large numbers of simulations.

    Args:
        params: The trainable parameters of the Equinox model.
        static: The static components of the Equinox model.
        ref_data (dict): A dictionary of reference data on the device.
        chunk_size (int): The number of simulations to process per chunk.

    Returns:
        dict: A dictionary containing error arrays and their mean values for
              each output variable and error type (e.g., 'L1_phi', 'avg_L2_c').
    """
    # 1. Reconstruct model and identify input/output keys
    model = eqx.combine(params, static)
    inp_idx, out_idx = model.inp_idx, model.out_idx
    input_keys = sorted(inp_idx.keys(), key=inp_idx.get)
    param_keys = [k for k in input_keys if k not in ["x", "t"]]
    output_keys = sorted(out_idx.keys(), key=out_idx.get)

    # 2. Set up spatial and temporal grids
    x_ref, t_ref = ref_data["x"], ref_data["t"]
    nx, nt = x_ref.shape[0], t_ref.shape[0]
    X_flat, T_flat = map(jnp.ravel, jnp.meshgrid(x_ref, t_ref, indexing="xy"))

    # 3. Prepare for scan loop over chunks
    num_sims = ref_data[param_keys[0]].shape[0] if param_keys else 1
    n_chunks = max(1, num_sims // chunk_size)

    def compute_chunk(carry, chunk_idx):
        """Processes one chunk of data to compute prediction errors."""
        start = chunk_idx * chunk_size
        param_chunks = tuple(lax.dynamic_slice(ref_data[k], (start,), (chunk_size,)) for k in param_keys)
        ref_output_chunks = tuple(lax.dynamic_slice(ref_data[k], (start, 0, 0), (chunk_size, nx, nt)) for k in output_keys)

        # Vmap the model evaluation over the chunk of parameters
        def eval_single_param_set(*params_single):
            preds_flat = vmap(model, in_axes=(0, 0, *(None,) * len(params_single)))(X_flat, T_flat, *params_single)
            return jax.tree.map(lambda arr: arr.reshape(nx, nt), preds_flat)

        pred_outputs = vmap(eval_single_param_set)(*param_chunks)
        # Vmap the error computation over the batch of predictions and references
        errs = [vmap(compute_errs)(ref, pred) for ref, pred in zip(ref_output_chunks, pred_outputs)]
        # Flatten the list of tuples for the scan's output
        return carry, tuple(err for triple in errs for err in triple)

    # 4. Execute the scan and process results
    _, results_per_chunk = lax.scan(compute_chunk, None, jnp.arange(n_chunks))
    flat_err_arrays = jax.tree_util.tree_map(lambda x: x.reshape(-1), results_per_chunk)

    # 5. Construct the final output dictionary
    output_dict = {}
    error_types = ["L1", "L2", "Linf"]
    for i, out_key in enumerate(output_keys):
        for j, err_type in enumerate(error_types):
            error_array = flat_err_arrays[i * len(error_types) + j]
            output_dict[f"{err_type}_{out_key}"] = error_array
            output_dict[f"avg_{err_type}_{out_key}"] = jnp.mean(error_array)

    return output_dict


def evaluate_model_performance(h5file, base_model, names=None, methods=None, sizes=None, models_dir="models", eval_chunk_size=20):
    """
    Orchestrates the evaluation of multiple models against reference data.

    This function finds models, loads reference data, determines an optimal
    chunk size, and then iterates through each model to call the core JIT-compiled
    evaluation function.

    Args:
        h5file (str): Path to the HDF5 file containing reference data.
        base_model (eqx.Module): A skeleton model instance for loading weights.
        names, methods, sizes: Filters passed to `find_model_paths`.
        models_dir (str): The root directory for models.
        eval_chunk_size (int): The desired chunk size for evaluation.

    Returns:
        dict: A nested dictionary summarizing the performance results for all
              evaluated models.
    """
    # --- Setup ---
    inp_idx, out_idx = base_model.inp_idx, base_model.out_idx
    param_keys = [k for k in sorted(inp_idx.keys(), key=inp_idx.get) if k not in ["x", "t"]]
    output_keys = sorted(out_idx.keys(), key=out_idx.get)

    print("=" * 60, f"\nStarting evaluation on: {h5file}")
    data = load_h5(h5file)
    ref_data_device = jax.tree_util.tree_map(jnp.asarray, data)
    n_sims = ref_data_device[param_keys[0]].shape[0] if param_keys else 1

    model_paths = find_model_paths(models_dir, names=names, methods=methods, sizes=sizes)
    if not model_paths:
        print("[WARNING] Found 0 models. Aborting.")
        return {}
    print(f"Found {len(model_paths)} models to evaluate.")

    # Adjust chunk size to be a divisor of n_sims for compatibility with lax.scan
    adjusted_chunk_size = next((i for i in range(eval_chunk_size, 0, -1) if n_sims % i == 0), 1)
    if adjusted_chunk_size != eval_chunk_size:
        print(f"[INFO] Adjusted chunk size from {eval_chunk_size} to {adjusted_chunk_size} for divisibility.")

    # --- Initialize the summary dictionary HERE ---
    summary = {}
    start_time, models_processed = time.perf_counter(), 0

    # --- Evaluation Loop ---
    for idx, model_info in enumerate(model_paths, start=1):
        if not model_info.get('model_path'):
            continue

        model_name_full = f"{model_info['name']}_{model_info['method']}_{model_info['size']}_{model_info['param_seed']}_{model_info['train_seed']}"
        print(f"Progress: {idx}/{len(model_paths)} | Evaluating: {model_name_full}...", end='\r', flush=True)

        try:
            # Load model and run evaluation
            model = load_model(base_model, model_info['model_path'])

            # (This is the fix from the previous error, which should be kept)
            model = eqx.tree_at(
                where=lambda m: m.span_pde,
                pytree=model,
                replace=base_model.span_pde
            )

            params, static = eqx.partition(model, eqx.is_array)
            results_dict = jax.device_get(
                evaluate_model_performance_single(params, static, ref_data_device, adjusted_chunk_size)
            )

            # Structure and store results
            model_summary = {p_key: data[p_key] for p_key in param_keys if p_key in data}
            for out_key in output_keys:
                model_summary[out_key] = {err: results_dict[f'{err}_{out_key}'] for err in ["L1", "L2", "Linf"]}
                model_summary[f'avg_loss_{out_key}'] = {err: results_dict[f'avg_{err}_{out_key}'] for err in ["L1", "L2", "Linf"]}

            summary_key = f"{model_info['name']}_{model_info['method']}_{model_info['size']}_{model_info['train_seed']}"
            summary[summary_key] = model_summary # Now this will work
            models_processed += 1
            del model, params
            gc.collect()
        except Exception as e:
            print(f"\n[ERROR] Failed processing {model_name_full}: {e}")

    # --- Final Report ---
    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 60, "\nFinished evaluation.")
    if models_processed > 0:
        print(f"Evaluated {models_processed} models in {total_time:.2f}s ({total_time / models_processed:.3f}s/model).")
    print("=" * 60)
    return summary


# ==============================================================================
# 4. Plotting Helper Functions
# ==============================================================================

def avg_err_metric_dotplot(summary, names, methods, sizes, plot_name, log_scale=False, out_dir="plots"):
    """
    Generates and saves dot plots comparing model error metrics across different seeds.

    This function takes a summary dictionary of model performance and creates
    dot plots for L1, L2, and Linf error metrics for variables 'phi' and 'c'.
    Each point on the plot represents the error from a single training seed,
    and a horizontal line indicates the mean error across seeds for that
    model configuration.

    Args:
        summary (dict): A dictionary where keys are model run identifiers and
            values are dictionaries containing performance metrics, including
            'avg_loss_phi' and 'avg_loss_c'.
        names (list[str]): A list of model setup names to include in the plot.
        methods (list[str]): A list of method/solver names to include.
        sizes (list[int]): A list of training data sizes to include.
        plot_name (str): The base name for the output plot files.
        log_scale (bool, optional): If True, the y-axis will be on a log scale.
            Defaults to False.
        out_dir (str, optional): The directory where plots will be saved.
            Defaults to "plots".
    """
    os.makedirs(out_dir, exist_ok=True)
    
    grouped = defaultdict(lambda: {"phi": {"L1": [], "L2": [], "Linf": []},
                                   "c":   {"L1": [], "L2": [], "Linf": []}})

    # --- MODIFICATION: Use robust parsing instead of a faulty regex ---
    for key, val in summary.items():
        # Parse the key from right to left by splitting
        try:
            # e.g., "train_setup_2_grid_inner_5_seed" ->
            # base = "train_setup_2_grid_inner"
            # size_str = "5"
            # train_seed = "seed"
            base, size_str, train_seed = key.rsplit('_', 2)
            size = int(size_str)
        except ValueError:
            # Skip keys that don't match the expected format
            continue

        # Now, find the method from the end of the 'base' string
        found_method = None
        name = None
        # Sort methods by length (desc) to handle cases like 'solver' and 'fast_solver'
        for method in sorted(methods, key=len, reverse=True):
            if base.endswith(f"_{method}"):
                found_method = method
                name = base.removesuffix(f"_{method}")
                break
        
        if not found_method:
            continue

        # Filter based on user-provided names, methods, and sizes
        if (names and name not in names) or \
           (methods and found_method not in methods) or \
           (sizes and size not in sizes):
            continue

        # The key for grouping models of the same configuration but different seeds
        group_key = f"{name}_{found_method}_{size}"
        
        for var in ["phi", "c"]:
            for metric in ["L1", "L2", "Linf"]:
                avg_loss = val[f'avg_loss_{var}'][metric]
                grouped[group_key][var][metric].append(avg_loss)

    # --- PLOTTING LOGIC (No changes needed here) ---
    
    sizes.sort()
    # Define the order and labels for the x-axis ticks
    # MODIFICATION: Use the correct method variable name
    group_order = [f"{name}_{method}_{size}" for name in names for size in sizes for method in methods]
    group_labels = [f"{method.upper()}_{size}" for name in names for size in sizes for method in methods]
    
    colors = {"phi": "tab:blue", "c": "tab:orange"}

    for metric in ["L1", "L2", "Linf"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle(f"{plot_name} ({metric} Error)", fontsize=16)

        for ax, var in zip(axes, ["phi", "c"]):
            has_data = False
            # Define x_ticks based on the a pre-defined group order
            x_ticks = {group: i for i, group in enumerate(group_order)}
            
            for group, data in grouped.items():
                if group in x_ticks:
                    i = x_ticks[group] # Get the pre-defined position for this group
                    if data[var][metric]:
                        y_values = data[var][metric]
                        x_values = np.full(len(y_values), i)
                        
                        ax.scatter(x_values, y_values, alpha=0.6, color=colors[var], label=var if not has_data else "")
                        
                        mean_across_seeds = np.mean(y_values)
                        ax.hlines(mean_across_seeds, i - 0.25, i + 0.25, colors=colors[var], linewidth=2.5)
                        
                        has_data = True

            ax.set_xticks(range(len(group_order)))
            ax.set_xticklabels(group_labels, rotation=45, ha='right')
            ax.set_title(f"Variable: {var}")
            ax.set_ylabel(f"Average {metric} Error")
            if log_scale:
                ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.5)
            if has_data:
                ax.legend()

        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        
        out_path = os.path.join(out_dir, f"{plot_name}_{metric}_dotplot.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {out_path}")

        # Choose which plot to show interactively
        if metric == "L2":
            plt.show()
        else:
            plt.close(fig)

def _load_and_process_loss_data(loss_path):
    """
    Loads and processes loss data from a single pytree file.

    This function attempts to load a pytree from the given path. It extracts
    various loss components ('ic', 'bc', 'ac', 'ch', 'data') and calculates their
    true geometric mean, provided all five components are present. It also
    extracts scatter loss data ('L2_phi', 'L2_c').

    Args:
        loss_path (str): The file path to the saved loss data.

    Returns:
        dict | None: A dictionary containing 'line' and 'scatter' loss data,
            or None if the file fails to be processed. The 'geom_loss' key
            is added to the 'line' dictionary if calculation is successful.
    """
    try:
        loss_data = load_pytree(loss_path)
        component_keys = ['ic', 'bc', 'ac', 'ch', 'data']
        line_losses = {k: loss_data.get(k) for k in component_keys}
        
        # --- NEW: Calculate true geometric mean based on user guarantee ---
        components = [np.asarray(line_losses.get(k)) for k in component_keys]
        
        # Check if all 5 components are present and valid before calculating
        if all(c is not None and c.ndim > 0 for c in components):
            stacked_components = np.stack(components)
            # Calculate product along the component axis, then take the 5th root
            line_losses['geom_loss'] = np.prod(stacked_components, axis=0) ** 0.2
        else:
            print(f"\n[INFO] Skipping geometric mean for {os.path.basename(loss_path)} due to missing components.")

        scatter_losses = {k: loss_data.get(k) for k in ['L2_phi', 'L2_c']}
        return {'line': line_losses, 'scatter': scatter_losses}
    except Exception as e:
        print(f"\n[WARNING] Failed to process {os.path.basename(loss_path)}: {e}")
        return None

def _setup_plot(title, ylabel, figsize=(15, 9)):
    """
    Initializes a Matplotlib figure and axes for plotting.

    Args:
        title (str): The title for the plot.
        ylabel (str): The label for the y-axis.
        figsize (tuple, optional): The size of the figure. Defaults to (15, 9).

    Returns:
        tuple: A tuple containing the Matplotlib figure and axes objects
            (fig, ax).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Epoch / Iteration", fontsize=12)
    return fig, ax

def _finalize_and_save_plot(fig, ax, save_path, log_scale, legend_kwargs=None, tight_layout_rect=None):
    """
    Applies final styling to a plot, adds a legend, and saves it to a file.

    Args:
        fig (matplotlib.figure.Figure): The figure object to finalize.
        ax (matplotlib.axes.Axes): The axes object to finalize.
        save_path (str): The full path where the plot image will be saved.
        log_scale (bool): If True, the y-axis is set to a logarithmic scale.
        legend_kwargs (dict, optional): Additional arguments to pass to the
            ax.legend() function. Defaults to None.
        tight_layout_rect (list, optional): The rectangle (left, bottom, right, top)
            for fig.tight_layout(). Defaults to None.
    """
    if log_scale:
        ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **(legend_kwargs or {"fontsize": 12}))

    fig.tight_layout(rect=tight_layout_rect)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

# ==============================================================================
# === Main Plotting Functions ==================================================
# ==============================================================================

def plot_loss_history(names, methods, sizes, models_dir="models", 
                      skip1=10, skip2=5, log_scale=True):
    """
    Generates and saves loss history plots for individual model runs.

    This function finds all model runs matching the specified criteria (names,
    methods, sizes) and generates a separate loss history plot for each one.
    Each plot shows the evolution of different loss components over training
    epochs/iterations.

    Args:
        names (list[str]): A list of model setup names to plot.
        methods (list[str]): A list of method/solver names to plot.
        sizes (list[int]): A list of training data sizes to plot.
        models_dir (str, optional): The base directory where model outputs are
            stored. Defaults to "models".
        skip1 (int, optional): The step size for subsampling line plot data to
            reduce plot density. Defaults to 10.
        skip2 (int, optional): The step size for subsampling scatter plot data.
            Defaults to 5.
        log_scale (bool, optional): If True, the y-axis of the plots will be on
            a log scale. Defaults to True.
    """
    sns.set_theme(style="whitegrid", palette="deep")
    print("=" * 60, "\nStarting individual loss history plotting...")
    model_paths_info = find_model_paths(models_dir, names=names, methods=methods, sizes=sizes)
    
    if not model_paths_info:
        print("[WARNING] Found 0 models. Aborting.")
        return

    print(f"Found {len(model_paths_info)} models. Generating plots...")
    start_time, plots_generated = time.perf_counter(), 0

    for idx, model_info in enumerate(model_paths_info, start=1):
        if not model_info['loss_path']: continue
        
        model_name = f"{model_info['name']}_{model_info['method']}_{model_info['size']}_{model_info['param_seed']}_{model_info['train_seed']}"
        print(f"Progress: {idx}/{len(model_paths_info)} | Processing: {model_name}...", end='\r', flush=True)
        
        losses = _load_and_process_loss_data(model_info['loss_path'])
        if not losses: continue

        title = f"Loss History: {model_name}" + (" (Log Scale)" if log_scale else "")
        ylabel = "Loss Value" + (" (Log Scale)" if log_scale else "")
        fig, ax = _setup_plot(title, ylabel)

        # Plot all line losses, including the new 'geom_loss'
        for name, data in losses['line'].items():
            if data is not None and data.ndim > 0:
                ax.plot(np.arange(len(data))[::skip1], data[::skip1], label=name, alpha=0.9, zorder=1)

        markers = {'L2_phi': 'o', 'L2_c': 'X'}
        for name, data in losses['scatter'].items():
            if data is not None and data.ndim > 0:
                valid_indices = np.where(~np.isnan(data))[0]
                if len(valid_indices) > 0:
                    plot_indices = valid_indices[::skip2]
                    ax.scatter(plot_indices, data[plot_indices], label=name, marker=markers[name], s=50, alpha=0.8, zorder=2)
        
        save_path = os.path.join(os.path.dirname(model_info['loss_path']), f"loss_history_{model_name}.png")
        _finalize_and_save_plot(fig, ax, save_path, log_scale)
        plt.close(fig)
        plots_generated += 1
            
    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 60, "\nFinished plotting.")
    if plots_generated > 0:
        print(f"Generated {plots_generated} plots in {total_time:.2f}s ({total_time / plots_generated:.3f}s/plot).")
    else:
        print("No plots were generated.")
    print("=" * 60)

def aggregated_loss_history(names, methods, sizes, models_dir, plot_name, out_dir="plots",
                            skip1=10, skip2=5, log_scale=True):
    """
    Generates a single plot showing the aggregated loss history for model families.

    This function groups models by their configuration (name, method, size) and
    calculates the average loss across different training seeds for each group.
    It then plots these averaged loss curves on a single figure for easy
    comparison between model families.

    Args:
        names (list[str]): A list of model setup names to include.
        methods (list[str]): A list of method/solver names to include.
        sizes (list[int]): A list of training data sizes to include.
        models_dir (str): The base directory where model outputs are stored.
        plot_name (str): The file name for the output plot (without extension).
        out_dir (str, optional): The directory where the plot will be saved.
            Defaults to "plots".
        skip1 (int, optional): The step size for subsampling line plot data.
            Defaults to 10.
        skip2 (int, optional): The step size for subsampling scatter plot data.
            Defaults to 5.
        log_scale (bool, optional): If True, the y-axis will be on a log scale.
            Defaults to True.
    """
    sns.set_theme(style="whitegrid", palette="deep")
    print("=" * 60, "\nStarting aggregated loss history plotting...")
    model_paths_info = find_model_paths(models_dir, names=names, methods=methods, sizes=sizes)
    
    if not model_paths_info:
        print("[WARNING] Found 0 models. Aborting.")
        return

    grouped_models = defaultdict(list)
    for info in model_paths_info:
        if info['loss_path']:
            grouped_models[f"{info['name']}_{info['method']}_{info['size']}"].append(info['loss_path'])

    print(f"Found {len(model_paths_info)} models across {len(grouped_models)} families.")

    title = f"Aggregated Loss History: {plot_name}" + (" (Log Scale)" if log_scale else "")
    ylabel = "Average Loss Value" + (" (Log Scale)" if log_scale else "")
    fig, ax = _setup_plot(title, ylabel, figsize=(16, 10))
    
    color_palette = sns.color_palette("husl", len(grouped_models))
    markers = {'L2_phi': 'o', 'L2_c': 'X'}

    for i, (family_key, paths) in enumerate(grouped_models.items()):
        # --- FINAL FIX: Use the base color directly. Do not manipulate it. ---
        base_color = color_palette[i]

        family_losses = [data for path in paths if (data := _load_and_process_loss_data(path))]
        if not family_losses: continue

        geom_loss_arrays = [d['line']['geom_loss'] for d in family_losses if d['line'].get('geom_loss') is not None]
        
        if geom_loss_arrays:
            min_len = min(len(arr) for arr in geom_loss_arrays)
            if min_len > 0:
                avg_geom_loss = np.nanmean([np.asarray(arr)[:min_len] for arr in geom_loss_arrays], axis=0)
                print(f"[PLOT] Plotting LINE for '{family_key} (Geom)' with {len(avg_geom_loss[::skip1])} points.")
                # Use the robust base_color
                ax.plot(np.arange(len(avg_geom_loss))[::skip1], avg_geom_loss[::skip1], label=f"{family_key} (Geom)", color=base_color, lw=2.5, zorder=1)
        else:
            print(f"[INFO] No 'geom_loss' data found for family '{family_key}'. Skipping line plot.")

        for loss_name in ['L2_phi', 'L2_c']:
            scatter_arrays = [d['scatter'][loss_name] for d in family_losses if d['scatter'].get(loss_name) is not None]
            if not scatter_arrays:
                continue

            min_len = min(len(arr) for arr in scatter_arrays)
            if min_len == 0: continue

            padded = [np.pad(np.asarray(a), (0, min_len - len(a)), 'constant', constant_values=np.nan) if len(a) < min_len else np.asarray(a)[:min_len] for a in scatter_arrays]
            
            avg_loss = np.nanmean(np.asarray(padded), axis=0)
            
            valid_indices = np.where(~np.isnan(avg_loss))[0]
            if len(valid_indices) > 0:
                plot_indices = valid_indices[::skip2]
                print(f"[PLOT] Plotting SCATTER for '{family_key} ({loss_name})' with {len(plot_indices)} points.")
                # Use the robust base_color
                ax.scatter(plot_indices, avg_loss[plot_indices], label=f"{family_key} ({loss_name})",
                           marker=markers[loss_name], color=base_color, s=60, alpha=0.9, edgecolor='black', linewidth=0.5, zorder=2)

    save_path = os.path.join(out_dir, f"{plot_name}.png")
    legend_kwargs = {"fontsize": 11, "title": "Model Family", "bbox_to_anchor": (1.04, 1), "loc": "upper left"}
    _finalize_and_save_plot(fig, ax, save_path, log_scale, legend_kwargs=legend_kwargs, tight_layout_rect=[0, 0, 0.85, 1])
    
    print("-" * 60, f"\nSaved aggregated plot to: {save_path}\n", "=" * 60)
    plt.show()
    plt.close(fig)