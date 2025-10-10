import numpy as np
from scipy import stats

def choose_default_scale(width_distr, scale):
    if scale is None:
        if width_distr == 'normal':
            scale = 0.1
        elif width_distr == 'uniform':
            scale = 0.4
        else:
            raise ValueError(f"Unknown width_distr: {width_distr}")
    return scale

def get_sampler(width_distr, rng):
    if rng is None:
        rng = np.random.default_rng()
    
    if width_distr == 'normal':
        sampler = lambda loc, scale, size: rng.normal(loc, scale, size)
    elif width_distr == 'uniform':
        sampler = lambda loc, scale, size: rng.uniform(loc-scale/2, loc+scale/2, size)
    else:
        raise ValueError(f"Unknown width_distr: {width_distr}")
    
    return sampler

def _is_scalar_like(c):
    # strings or 3/4-length tuples/lists are scalar colors
    if isinstance(c, str):
        return True
    if isinstance(c, (tuple, list)) and len(c) in (3, 4) and all(
        isinstance(v, (int, float)) for v in c
    ):
        return True
    return False

def _resolve_variable(v, data):
        """ Resolve the variable v to an array, using data if v is a string."""
        if v is None:
            return None
        if isinstance(v, str):
            if data is None:
                raise ValueError(f"`{v}` was given as a column name, but `data` is None.")
            return data[v].to_numpy()
        # assume already an array-like
        return np.asarray(v)

def _looks_numeric(a):
        """ Heuristic to decide if an array-like is kinda numeric. """
        if a is None:
            return False
        try:
            # numeric if coercible; NaNs allowed
            return np.issubdtype(np.asarray(a).dtype, np.number)
        except Exception:
            return False

def _infer_orient(x_vals, y_vals, orient):
    """ Infer orientation if orient is None, otherwise validate it."""
    # infer orientation if None
    if orient is None:
        if x_vals is None and y_vals is not None:
            # Only y provided → categorical on x
            orient = "x"
        elif y_vals is None and x_vals is not None:
            # Only x provided → categorical on y
            orient = "y"
        else:
            # both provided → pick based on numeric vs categorical
            if _looks_numeric(y_vals) and not _looks_numeric(x_vals):
                orient = "x"
            elif _looks_numeric(x_vals) and not _looks_numeric(y_vals):
                orient = "y"
            else:
                # fallback (seaborn defaults to categorical on x)
                orient = "x"
    else:
        orient = orient.lower()
        # alias 'h'/'v' into 'x'/'y'
        if orient == 'h':
            orient = 'x'
        elif orient == 'v':
            orient = 'y'
        # validate
        if orient not in ("x", "y"):
            raise ValueError("orient must be 'h', 'v', 'x', 'y', or None")

    return orient

def _dodge_offsets(n_levels, dodge):
    if n_levels <= 1:
        return [0.0]
    # if not dodging, all offsets are zero
    if not dodge:
        return [0.0] * n_levels
    # if dodging, use default width unless dodge is float, in which case use that
    if isinstance(dodge, float):
        width = dodge
    else:
        width = 0.3

    # centered positions across [-width/2, width/2]
    return np.linspace(-width / 2.0, width / 2.0, n_levels)

def _record_drawn(clouds, cloud, boxes, box):
    if cloud is not None:
        clouds.append(cloud)
        boxes.append(box)
        return True
    return False

def choose_width_function(width_method):
    if width_method == 'histogram':
        return histogram_function
    elif width_method == 'kde':
        return kde_function
    else:
        raise ValueError(f"Unknown width_method: {width_method}")

def histogram_function(data, **hist_args):
    if 'bins' not in hist_args:
        hist_args['bins'] = 'auto'

    hist, bin_edges = np.histogram(data, **hist_args)
    max_val = hist.max()

    def hist_at_point(x):
        # Use vectorized calls so we can handle x being an array
        x = np.asarray(x)
        idx = np.searchsorted(bin_edges, x, side='right') - 1

        # Where x is exactly the last edge, assign to last bin
        idx = np.where(x == bin_edges[-1], len(hist) - 1, idx)
        
        # If x is outside the bin range, return 0
        out_of_bounds = (idx < 0) | (idx > len(hist) - 1)

        # Temp results where we clip values out of bounds
        result = hist[np.clip(idx, 0, len(hist) - 1)] / max_val if max_val > 0 else 0

        # Replace out-of-bounds results with 0
        result = np.where(out_of_bounds, 0, result)

        return result if result.shape != () else result.item()

    return hist_at_point

def kde_function(data, **kde_args):
    kde = stats.gaussian_kde(data, **kde_args) if len(data)>1 else (lambda y: np.ones_like(y, dtype=float))
    max_val = kde(data).max()

    def kde_at_point(x):
        return kde(x) / max_val if max_val > 0 else np.zeros_like(x, dtype=float)
    
    return kde_at_point