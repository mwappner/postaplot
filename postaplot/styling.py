""" Styling utilities for postaplot. Includes functions to parse hue choice (incl.
discrete/continuous, mapped palletes, colormaps, etc.), and to normalize plot() 
kwargs to scatter() kwargs, including handling of per-point colors."""

from warnings import warn
from typing import Mapping

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import Normalize

from .core import _is_scalar_like, _looks_numeric

def _marker_param_setter(params, key, val, alt_val=None):
    """ If key does not exist yet, set params[key] = val if val is not None, else alt_val."""
    if key not in params:
        if val is not None:
            params[key] = val
        else:
            params[key] = alt_val

def _alias_scatter_kwargs(plot_kw):
    """
    Normalize plot() / Line2D-style marker kwargs to scatter()-style kwargs.

    - markersize/ms (points)  -> s (points^2)   [scatter expects area]
    - markeredgewidth/mew/lw  -> linewidths
    - markeredgecolor/mec     -> edgecolors
    - markerfacecolor/mfc     -> color (iff no per-point colors are being passed)
    - edgecolor               -> edgecolors
    - facecolor               -> color (iff no per-point colors)
    - Keeps marker, alpha, rasterized, etc. as is.

    Special cases:
    - we try to map size into s, but if it results in NaN, we ignore it and use the 
        default value
    - 'hollow=True' sets facecolors='none' and overrides any facecolor/mfc setting.
    - If facecolors is set:
        - if edgecolors is also set, per-point colors are ignored (assumed intentional)
        - else, if per-point colors are set, they are applied to edges
        - else, edgecolors defaults to facecolors (default Matplotlib behavior)
    """
    pk = dict(plot_kw)  # don't mutate caller's dict

    # Hollow is an alias for no facecolor, warn if both were specified
    hollow = pk.pop("hollow", False)
    if hollow:
        # if facecolor/mfc was given, warn and ignore it
        if "markerfacecolor" in pk or "mfc" in pk or 'facecolors' in pk:
            warn("Warning: 'hollow=True' overrides any 'facecolors/markerfacecolor'/'mfc' setting.")
            pk.pop("markerfacecolor", None)
            pk.pop("mfc", None)
        
        # set facecolors to 'none', we'll handle per-point color later
        pk["facecolors"] = "none"

    # --- size: markersize (points) -> s (points^2)
    ms = pk.pop("markersize", pk.pop("ms", None)) # 's' takes presedence 
    try:
        ms_arr = np.asarray(ms, dtype=float) # in case list-like, so we can square it
        s = ms_arr ** 2  # scatter uses area in pt^2
        if not np.isfinite(s).all(): # some values map to NaN, so nothing gets plotted
            s = None
    except Exception:
        s=None
        warn(f"Warning: couldn't interpret markersize/ms={ms!r}; ignoring.")
        # if something odd, just drop-through (scatter will use default)
        pass
    _marker_param_setter(pk, "s", s)

    # --- edge width: markeredgewidth/mew/linewidth/lw -> linewidths
    mew = pk.pop("markeredgewidth", pk.pop("mew", None)) 
    lw  = pk.pop("linewidth", pk.pop("lw", None))
    _marker_param_setter(pk, "linewidths", mew, lw)

    # --- edge color: markeredgecolor/mec/edgecolor -> edgecolors
    mec = pk.pop("markeredgecolor", pk.pop("mec", None))
    ec  = pk.pop("edgecolor", None) 
    _marker_param_setter(pk, "edgecolors", mec, ec)

    # --- face color: markerfacecolor/mfc/facecolor -> color (only if not per-point)
    #                                               -> facecolors (if 'none' for hollow)
    mfc = pk.pop("markerfacecolor", pk.pop("mfc", None))
    fc  = pk.pop("facecolor", None) 
    _marker_param_setter(pk, "facecolors", mfc, fc)
    
    # if the user have both marker face and edge colors, assume they know what they're
    # doing and ignore the per-point colors given via 'color='
    if pk['facecolors'] is not None:
        if pk.get('edgecolors', None) is not None:
            pk.pop('c', None)
            pk.pop('color', None)
        else:
            # if colors were given via 'c=' but no edgecolor was given, move from 'c' to 'color'
            # so matplotlib doesn't warn a conflict between 'c' and 'edgecolors' and the
            # colors apply to edges
            if 'c' in pk:
                pk['color'] = pk.pop('c')

    return pk


def choose_color_plotting_mode(plot_kw, vert):
    color = plot_kw.pop("color", None)  # may be scalar, list-of-N, or (N,4) RGBA
    scatter_kwargs = dict(**plot_kw)

    if color is None:
        # nothing to pass; let Matplotlib pick next color in the cycle
        pass
    else:
        # normalize into one of:
        #  - scalar:   use color=...
        #  - length-N: use c=...
        #  - (N,3/4):  use c=... (ensure correct orientation)
        try:
            col = np.asarray(color)
        except Exception:
            # non-array-like: treat as scalar
            scatter_kwargs["color"] = color
        else:
            N = len(vert) # number of points being plotted

            if col.ndim == 0 or _is_scalar_like(color):
                # a single color → use 'color=' mode
                scatter_kwargs["color"] = color
            elif col.ndim == 1 and col.shape[0] == N:
                # one value per point (e.g., list of hex strings or numbers)
                scatter_kwargs["c"] = color
            elif col.ndim == 2:
                # accept (N,3) / (N,4); if (3/4, N), transpose
                if col.shape[0] in (3, 4) and col.shape[1] == N:
                    col = col.T
                # if now (N,3) or (N,4), use 'c=' mode
                if col.shape[0] == N and col.shape[1] in (3, 4):
                    scatter_kwargs["c"] = col
                else:
                    # fallback: treat as scalar color if it quacks like one
                    scatter_kwargs["color"] = color
            else:
                # strange shape → fallback
                scatter_kwargs["color"] = color
    return scatter_kwargs


def _resolve_hue_mapping(hue_vals, hue_order, palette, hue_norm=None):
    """Resolve hue semantics and colors.

    Returns:
        is_continuous_hue : bool
        hue_levels        : list (if discrete) or []
        pal_map           : dict level -> color (if discrete) or {}
        cmap              : Colormap or None (if continuous)
        norm              : Normalize or None (if continuous)

    NOTE:
        This is a simplified version of seaborn's internal _resolve_hue_mapping or something 
        like that, built with help from ChatGPT. 
        Since I built this, I moved the plotting engine from plt.plot to plt.scatter and updated 
        this function accordingly, which makes plotting mapped values much easier by passing 
        an array and a colormap/norm directly, but I'm not gonna rewrite this to exploit that
        until I have a good reason to.
    """
    # Defaults, if no hue
    is_continuous_hue = False
    hue_levels = []
    pal_map = {}
    cmap = None
    norm = None

    if hue_vals is None:
        return is_continuous_hue, hue_levels, pal_map, cmap, norm

    hv = np.asarray(hue_vals)
    is_numeric = _looks_numeric(hv)
    if is_numeric:
        # heuristic similar to seaborn: many unique numeric values -> continuous
        nunique = np.unique(hv).size
        is_continuous_hue = nunique > 8

    if is_continuous_hue:
        # CONTINUOUS HUE → use a colormap + normalizer
        if palette is None or isinstance(palette, (list, tuple, Mapping)):
            # ignore discrete palettes; use Matplotlib default cmap
            default_cmap_name = mpl.rcParams.get("image.cmap", "viridis")
            cmap = mpl.colormaps.get_cmap(default_cmap_name)
        else:
            # palette is a colormap name
            cmap = mpl.colormaps.get_cmap(palette)

        # handle normalization of continuous hue
        if hue_norm is None:
            vmin = float(np.nanmin(hv))
            vmax = float(np.nanmax(hv))
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        elif isinstance(hue_norm, Normalize):
            norm = hue_norm
        else:
            vmin, vmax = hue_norm
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    else:
        # DISCRETE HUE → resolve levels + color mapping
        hue_levels = pd.unique(hv) if hue_order is None else list(hue_order)
        hue_levels = [h for h in hue_levels if np.any(hv == h)]

        # if no palette, use default color cycle
        if palette is None:
            cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3"])
            pal_map = {h: cycle[i % len(cycle)] for i, h in enumerate(hue_levels)}

        # if palette is a mapping, cast it to dict and use directly and hope the keys are right
        elif isinstance(palette, Mapping):
            pal_map = dict(palette)
        
        # if palette is a list/tuple, map levels to colors in order
        elif isinstance(palette, (list, tuple)):
            if len(palette) < len(hue_levels):
                raise ValueError("Palette list is shorter than the number of hue levels.")
            pal_map = {h: palette[i] for i, h in enumerate(hue_levels)}
        
        # if palette is a string, interpret as matplotlib colormap name and hope it's valid
        else:
            cm = mpl.colormaps.get_cmap(palette)
            denom = max(1, len(hue_levels) - 1)
            pal_map = {h: cm(i / denom) for i, h in enumerate(hue_levels)}

    return is_continuous_hue, hue_levels, pal_map, cmap, norm

