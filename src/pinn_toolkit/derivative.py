import jax
import jax.numpy as jnp
import re
from jax import grad, vmap
from collections import defaultdict

class Derivative:
    """
    Generic derivative generator for JAX-based models.

    Attributes:
        inp_idx: Maps input names to argument indices.
        out_idx: Maps output names to output indices.
        phys_span: Physical variable ranges.
        norm_span: Normalized variable ranges.
        norm_coefs: Scaling factors for normalized derivatives.
    """

    def __init__(self, inp_idx, out_idx, phys_span, norm_span):
        """
        Initialize Derivative instance.

        Args:
            inp_idx: Mapping of input variable names to indices.
            out_idx: Mapping of output variable names to indices.
            phys_span: Physical variable ranges (min,max) for scaling.
            norm_span: Normalized variable ranges (min,max) for scaling.
        """
        self.inp_idx = inp_idx
        self.out_idx = out_idx
        self.phys_span = phys_span
        self.norm_span = norm_span
        self.norm_coefs = {name: self._get_coef(norm_span[name], phys_span[name])
                           for name in inp_idx}

    def _get_coef(self, norm_range, phys_range):
        """
        Compute scaling factor from normalized to physical units.

        Args:
            norm_range: (min,max) in normalized space.
            phys_range: (min,max) in physical space.

        Returns:
            float: scaling factor.
        """
        if phys_range[1] == phys_range[0]:
            return 1.0
        return (norm_range[1] - norm_range[0]) / (phys_range[1] - phys_range[0])

    def _parse_name(self, name_str):
        """
        Parse derivative identifier string.

        Args:
            name_str: e.g., 'phi_x2_t'.

        Returns:
            output name, derivative orders per variable as dict.
        """
        parts = name_str.split('_')
        out_name = parts[0]
        deriv_order = defaultdict(int)
        pattern = re.compile(r"([a-zA-Z]+)(\d*)|(\d+)([a-zA-Z]+)")

        for token in parts[1:]:
            match = pattern.fullmatch(token)
            if not match:
                raise ValueError(f"Invalid derivative token '{token}' in '{name_str}'")
            var = match.group(1) or match.group(4)
            pow_str = match.group(2) or match.group(3)
            order = int(pow_str) if pow_str else 1
            deriv_order[var] += order

        return out_name, dict(deriv_order)

    def create_deriv_fn(self, name_str):
        """
        Dynamically create a derivative function and attach it.

        Args:
            name_str: derivative identifier, e.g., 'phi_x2_t'.
        """
        out_name, deriv_order = self._parse_name(name_str)

        def base_fn(model, *args):
            """Scalar output function for one sample."""
            out = model(*args)
            return out[self.out_idx[out_name]]

        grad_fn = base_fn
        for var, order in deriv_order.items():
            argnum = self.inp_idx[var] + 1  # skip model arg
            for _ in range(order):
                grad_fn = grad(grad_fn, argnums=argnum)

        scale = jnp.prod(jnp.array([self.norm_coefs[v] ** o for v, o in deriv_order.items()]))

        def deriv_fn_single(model, *args):
            """Compute derivative for a single sample."""
            return scale * grad_fn(model, *args)

        # Wrapped function that auto-vectorizes if inputs are arrays
        def deriv_fn(model, *args):
            # Convert to array and inspect dims
            arrs = [jnp.asarray(a) for a in args]
            # If all scalars, just compute directly
            if all(jnp.ndim(a) ==0 for a in arrs):
                return deriv_fn_single(model, *arrs)
            else:
                in_axes = (None,) + tuple(None if jnp.ndim(a) == 0 else 0 for a in arrs)
                return jax.vmap(deriv_fn_single, in_axes = in_axes)(model, *arrs)

        # Attach to instance
        setattr(self, name_str, deriv_fn)

    def evaluate(self, model, *args, function_names: list):
        """
        Compute multiple registered derivatives.
        Returns a dict mapping name -> array.
        """
        results = {}
        for name in function_names:
            fn = getattr(self, name, None)
            results[name] = fn(model, *args) if fn else jnp.array([])
        return results
