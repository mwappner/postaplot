import numpy as np

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
