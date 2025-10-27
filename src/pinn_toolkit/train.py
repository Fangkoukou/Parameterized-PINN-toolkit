"""
Stateless JAX/Equinox Training Module.

This module provides a generic, stateless, and JIT-compatible training loop
for Equinox models. It is designed with JAX best practices in mind, using
`lax.scan` for the main loop to ensure high performance.

The design is flexible, allowing for dynamic resampling of training data,
re-weighting of loss components, and periodic validation, all controlled by
schedule frequencies.
"""

# ==============================================================================
# 1. Imports
# ==============================================================================

import time
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax.experimental import io_callback

# ==============================================================================
# 2. Host-Side Logging Utilities
# ==============================================================================

# Global variables for tracking training time.
# These are managed on the host and are not part of the JIT-compiled computation.
LOG_TIMES: List[Tuple[int, float, float]] = []
LOG_START: float = 0.0
LOG_LAST: float = 0.0


def _log_on_host(step, total_loss, loss_vals, weight_vals, *, keys_tuple):
    """
    The actual printing function that runs on the Python host.

    This function is called from the JIT-compiled code via `io_callback`.
    It formats and prints a detailed log of the training progress for a given step.
    """
    global LOG_START, LOG_LAST, LOG_TIMES
    now = time.time()
    total_elapsed = now - LOG_START
    epoch_elapsed = now - LOG_LAST
    LOG_TIMES.append((int(step), float(total_elapsed), float(epoch_elapsed)))
    LOG_LAST = now

    # --- Formatting ---
    keys = list(keys_tuple)
    col_width = 12
    formatted_keys = " | ".join([f"{k:^{col_width}}" for k in keys])

    def format_array_line(arr):
        formatted_numbers = [f"{x:<{col_width}.4e}" for x in arr]
        return " | ".join(formatted_numbers)

    weights_line = format_array_line(weight_vals)
    losses_line = format_array_line(loss_vals)
    prod_line = format_array_line(loss_vals * weight_vals)

    header = (
        f"Step {step:<6} total_loss = {total_loss:<10.4f}  "
        f"epoch_elapsed = {epoch_elapsed:.3f}s  "
        f"total_elapsed = {total_elapsed:.3f}s"
    )

    # --- Printing ---
    print("=" * len(header))
    print(header)
    print(
        f"Keys   : {formatted_keys}\n"
        f"Weights: {weights_line}\n"
        f"Losses : {losses_line}\n"
        f"W * L  : {prod_line}"
    )


# ==============================================================================
# 3. Training Class
# ==============================================================================


class Train:
    """
    A stateless utility class for orchestrating JAX-based training.

    This class acts as a namespace for grouping related training functions.
    It does not hold any state itself; all state is managed explicitly through
    function arguments and the `carry` mechanism in `lax.scan`.
    """

    @staticmethod
    def _log_impl(step, loss_dict, weight_dict, *, log_keys_order: Tuple[str, ...]):
        """
        Prepares data and triggers the host-side logging callback.

        This internal method gathers the relevant loss and weight values,
        calculates the total weighted loss, and then uses `io_callback` to
        schedule `_log_on_host` for execution outside the JIT context.
        """
        # Determine the final order of keys to log
        initial_keys = log_keys_order or list(loss_dict.keys())
        final_keys = [key for key in initial_keys if key in loss_dict]

        # Gather corresponding values
        loss_vals = jnp.array([loss_dict[key] for key in final_keys])
        weight_vals = jnp.array([weight_dict.get(key, 1.0) for key in final_keys])
        total_loss = jnp.sum(loss_vals * weight_vals)

        # Create a partial function with the keys tuple "frozen"
        host_fn_with_keys = partial(_log_on_host, keys_tuple=tuple(final_keys))

        # Schedule the host-side function call
        io_callback(
            host_fn_with_keys,
            None,  # No result is returned from the host to JAX
            step,
            total_loss,
            loss_vals,
            weight_vals,
            ordered=True  # Ensures logs print in the correct order
        )

    @staticmethod
    def train_step(carry, step, static, optimizer, update_input, update_weight,
                   training_loss_fn, validation_fn, log_keys_order,
                   sp1, sp2, sp3, sp4, num_data, P_model, train_data):
        """
        Performs a single training step within the `lax.scan` loop.

        This function encapsulates the logic for one iteration: resampling data,
        updating weights, computing gradients, applying optimizer updates,
        tracking the best model, and conditionally logging/validating.

        Args:
            carry (tuple): The state carried over from the previous step. Contains:
                - key: JAX PRNG key.
                - sample: The current batch of training data.
                - weight_dict: The current dictionary of loss weights.
                - params: The model's trainable parameters.
                - opt_state: The optimizer's state.
                - best_loss: The best validation loss seen so far.
                - best_params: The parameters corresponding to the best loss.
            step (int): The current training step index.
            static, optimizer, ...: Static arguments that do not change per step.
            sp1, sp2, sp3, sp4 (int): Schedule periods for resampling, re-weighting,
                                     logging, and validation, respectively.

        Returns:
            A tuple containing the updated `carry` for the next step and a
            dictionary of losses for the current step's history.
        """
        # 1. Unpack state and update PRNG key
        key, sample, weight_dict, params, opt_state, best_loss, best_params = carry
        key1, key2, key3 = jax.random.split(key, 3)

        # 2. Conditionally resample training data every `sp1` steps
        sample = lax.cond(
            (step > 0) & (step % sp1 == 0),
            lambda k: update_input(k, P_model, train_data, num_data),
            lambda _: sample,  # Pass through existing sample
            operand=key1,
        )

        # 3. Conditionally update loss weights every `sp2` steps
        weight_dict = lax.cond(
            (step > 0) & (step % sp2 == 0),
            lambda k: update_weight(k, params, static, sample),
            lambda _: weight_dict,  # Pass through existing weights
            operand=key2,
        )

        # 4. Compute training loss and gradients
        (loss, train_loss_dict), grads = jax.value_and_grad(
            training_loss_fn, has_aux=True
        )(params, static, sample, weight_dict)

        # 5. Apply optimizer updates
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # 6. Track best parameters based on the geometric mean of key losses
        # This prevents one loss component from dominating the "best" metric.
        epsilon = 1e-12
        keys_for_gmean = ['ic', 'bc', 'ac', 'ch', 'data']
        log_losses = jnp.log(
            jnp.asarray([train_loss_dict.get(k, 0.0) for k in keys_for_gmean]) + epsilon
        )
        geometric_mean_loss = jnp.exp(jnp.mean(log_losses))

        best_loss, best_params = lax.cond(
            geometric_mean_loss < best_loss,
            lambda: (geometric_mean_loss, params),  # Update best
            lambda: (best_loss, best_params),     # Keep existing
        )

        # 7. Conditionally compute validation loss and log
        should_validate = (step % sp4 == 0) | (step % sp3 == 0)

        # This JAX-friendly structure ensures the output PyTree is consistent,
        # which is a requirement for `lax.cond`.
        def compute_and_merge_validation(t_loss_dict):
            combined = t_loss_dict.copy()
            combined.update(validation_fn(params, static))
            return combined

        def add_nan_placeholders(t_loss_dict):
            combined = t_loss_dict.copy()
            combined.update({'L2_phi': jnp.nan, 'L2_c': jnp.nan})
            return combined

        loss_dict_for_history = lax.cond(
            should_validate,
            compute_and_merge_validation,
            add_nan_placeholders,
            operand=train_loss_dict
        )

        # Trigger logging callback every `sp3` steps
        lax.cond(
            step % sp3 == 0,
            lambda _: Train._log_impl(step, loss_dict_for_history, weight_dict, log_keys_order=log_keys_order),
            lambda _: None,
            operand=None
        )

        # 8. Return updated state and current step's loss history
        new_carry = (key3, sample, weight_dict, params, opt_state, best_loss, best_params)
        return new_carry, loss_dict_for_history

    @staticmethod
    def train(total_steps, sp1, sp2, sp3, sp4, num_data, static, optimizer,
              update_input, update_weight, loss_fn, validation_fn,
              log_keys_order, P_model, train_data, carry):
        """
        Encapsulate the entire training process using a `lax.scan` loop.

        This is the main public-facing function. It sets up the training step,
        JIT-compiles the entire loop, executes it, and returns the final results.

        Args:
            total_steps (int): The total number of training steps to run.
            sp1, sp2, sp3, sp4 (int): Frequencies for resampling, re-weighting,
                                     logging, and validation.
            ... (various): Callbacks, model components, and data needed for training.
            carry (tuple): The initial state for the training loop.

        Returns:
            A tuple containing the final `carry` (final state) and the
            `loss_history` (a PyTree of losses recorded at each step).
        """
        global LOG_START, LOG_LAST
        LOG_START = time.time()
        LOG_LAST = LOG_START

        # Use partial to "freeze" the static arguments for the step function,
        # making its signature compatible with `lax.scan`.
        step_fn = partial(
            Train.train_step,
            static=static,
            optimizer=optimizer,
            update_input=update_input,
            update_weight=update_weight,
            training_loss_fn=loss_fn,
            validation_fn=validation_fn,
            log_keys_order=log_keys_order,
            sp1=sp1, sp2=sp2, sp3=sp3, sp4=sp4,
            num_data=num_data,
            P_model=P_model,
            train_data=train_data
        )

        # JIT-compile the entire scan operation for maximum performance.
        @jax.jit
        def _run_scan(initial_carry):
            return lax.scan(step_fn, initial_carry, xs=jnp.arange(total_steps + 1))

        # Execute the training loop
        final_carry, loss_history = _run_scan(carry)

        print(f"The training time is {(time.time() - LOG_START) / 60:.2f} minutes")
        return final_carry, loss_history