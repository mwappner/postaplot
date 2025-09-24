from __future__ import annotations
from typing import Literal, Mapping, Sequence, Optional, Any, Union
from matplotlib.colors import Colormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.axes import Axes

def build_discrete_legend_handles(
    hue_levels: Sequence[Any],
    pal_map: Mapping[Any, Any],
    *,
    hollow: bool,
    marker: str = "o",
    markersize_pt: float = 6.0,
    linewidth: Optional[float] = None,
) -> list[Line2D]:
    """Create proxy Line2D handles for a discrete-hue legend."""
    face = "none" if hollow else None  # None → use line's color as facecolor
    edge_linewidth = 0.8 if (linewidth is None and hollow) else linewidth

    handles = []
    for lvl in hue_levels:
        col = pal_map.get(lvl, "C0")
        # For hollow: facecolor 'none', edgecolor col
        # For filled: facecolor col, edgecolor col
        h = Line2D(
            [0], [0],
            marker=marker, linestyle="",
            markersize=markersize_pt,
            markerfacecolor=(face if hollow else col),
            markeredgecolor=col,
            markeredgewidth=(edge_linewidth if edge_linewidth is not None else 0.8 if hollow else 0.0),
            color=col,  # affects face when not explicitly set
            label=str(lvl),
        )
        handles.append(h)
    return handles

def add_discrete_legend(
    ax,
    hue_title: Optional[str],
    hue_levels: Sequence[Any],
    pal_map: Mapping[Any, Any],
    *,
    hollow: bool,
    marker: str = "o",
    markersize_pt: float = 6.0,
    linewidth: Optional[float] = None,
    loc: str = "best",
):
    handles = build_discrete_legend_handles(
        hue_levels, pal_map,
        hollow=hollow, marker=marker, markersize_pt=markersize_pt, linewidth=linewidth
    )
    ax.legend(handles=handles, title=hue_title, loc=loc)

def add_continuous_colorbar(
    ax,
    cmap,
    norm,
    *,
    label: Optional[str] = None,
    orientation: str = "vertical",
):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # mpl >=3.8 ok; keeps older versions happy
    cbar = ax.figure.colorbar(sm, ax=ax, orientation=orientation)
    if label:
        cbar.set_label(label)
    return cbar

def _resolve_hue_label(hue) -> str | None:
    # If user gave a column name
    if isinstance(hue, str):
        return hue
    # If user gave a pandas Series/Index with a name
    try:
        if hue.name is not None: # if hue is Series or index, has name, else Exception
            return str(hue.name)
    except Exception:
        pass
    return None


def add_legend_or_colorbar(
        ax: Axes, 
        hue: Optional[Union[str, Sequence]],
        plot_kw: dict,
        show_reference:  Union[bool, Literal['legend','colorbar','auto']],
        is_continuous_hue: bool,
        hue_levels: Sequence[Any],
        pal_map: Mapping[Any, Any],
        cmap: Optional[Union[str, Colormap]] = None,
        norm: Optional[Normalize] = None
):
    """Add a legend or colorbar to the given Axes based on the hue variable."""

    # return early if no hue or no reference wanted
    if hue is None or not show_reference:
        return

    # decide what type of reference to show
    if show_reference is True:
        show_reference = 'auto'
    if show_reference not in ('legend', 'colorbar', 'auto'):
        raise ValueError(f"Invalid value for show_reference: {show_reference!r}. Must be 'legend', 'colorbar', 'auto', or a boolean.")
    if show_reference == 'auto':
        show_reference = 'colorbar' if is_continuous_hue else 'legend'

    # Decide if markers are hollow for legend appearance
    # Prefer the already-aliased kwargs if you expose them;
    # otherwise infer from user kwargs passed to kde_scatter:
    hollow_flag = False
    if plot_kw.get("hollow", False):
        hollow_flag = True
    # also treat explicit facecolors='none' / mfc='none' as hollow intent
    if str(plot_kw.get("markerfacecolor", plot_kw.get("mfc", ""))).lower() == "none":
        hollow_flag = True
    if str(plot_kw.get("facecolor", plot_kw.get("facecolors", ""))).lower() == "none":
        hollow_flag = True

    # Marker aesthetics for legend (rough guesses from kwargs)
    marker = plot_kw.get("marker", "o")
    # If user passed size via s/ms/markersize, try to show similar legend size
    ms_pt = 6.0
    if "markersize" in plot_kw:
        ms_pt = float(plot_kw["markersize"])
    elif "ms" in plot_kw:
        ms_pt = float(plot_kw["ms"])
    elif "s" in plot_kw:
        # s is area pt^2 → approximate radius scaling
        try:
            import numpy as np
            s_val = plot_kw["s"]
            s0 = float(np.atleast_1d(s_val)[0])
            ms_pt = (s0 ** 0.5)
        except Exception:
            pass

    # linewidth estimate for legend stroke
    lw_est = plot_kw.get("markeredgewidth", plot_kw.get("mew", plot_kw.get("linewidths", None)))
    hue_label = _resolve_hue_label(hue)
    
    if show_reference == 'colorbar':
        add_continuous_colorbar(ax, cmap, norm, label=hue_label)
    else:
        add_discrete_legend(
            ax, hue_title=hue_label, hue_levels=hue_levels, pal_map=pal_map,
            hollow=hollow_flag, marker=marker, markersize_pt=ms_pt, linewidth=lw_est
        )