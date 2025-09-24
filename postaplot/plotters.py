import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from typing import Literal, Optional, Sequence, Mapping, Union, Callable
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.collections import Collection

from .core import _resolve_variable, _infer_orient, get_sampler, choose_default_scale, _dodge_offsets
from .styling import _resolve_hue_mapping, choose_color_plotting_mode, _alias_scatter_kwargs
from .legend import add_legend_or_colorbar

def postaplot_engine(
    loc: Union[float, int, np.integer, np.floating],
    series: Union[Sequence, np.ndarray, pd.Series],
    scale: Optional[float] = None,
    ax: Optional[Axes] = None,
    alpha: float = 0.4,
    rasterized: Optional[bool] = None,
    width_distr: str = 'normal',
    orientation: str = 'vertical',
    rng: Optional[np.random.Generator] = None,
    bw_method: Optional[Union[str, float, Callable]] = None,
    **plot_kw,
) -> Optional[Collection]:
    """
    Make a categorical scatter plot with jitter of the data points in series at
    location loc.
    
    Assuming orientation='vertical', a cloud of points is generated around the
    vertical line x=loc. The vertical coordinate of each point in the cloud is
    given by the value of series. The horizontal coordinate is randomly offset
    around loc. The offset ammount is randomly chosen from a distribution with
    a width that reflects the value of the estimated density function of series
    at that point. The density function of series is estimated using a gaussian
    kernel density estimator. The overall effect is that, for sufficiently large
    samplesizes, the scattered cloud has a shape that resembles that of the
    (estimated) density function.

    Parameters
    ----------
    loc : float
        Horizontal coordinate around which the scatter cloud will generate.
    series : array-like
        Data to generate the scatter plot.
    scale : float or None, optional
        Decides how wide each scattered cloud should be. Assuming this function 
        will be called multiple times and that the values of i will be 
        increasing integers each time, 0.1 is a good value to generate decently
        sized clouds with good spacing between neighboring clouds. For a cloud
        with width at each point is chosen from a normal distribution (default), 
        it's hard to estimate the aparent final width of the cloud, but a good rule 
        of thumb is that the cloud will be about 4×scale wide. If None, scale is 
        set to 0.1 for a normal distribution and 0.4 for a uniform distribution. The
        default is None.
    ax : matplotlib Axes or None, optional
        Axes onto which to plot the scatter plot. If ax is None, the current 
        active axis is selected. This does not open a figure if none is open,
        so you will have to do that yourself. The default is None.
    alpha : float in [0, 1], optional
        Alpha value  (transparency) of the plotted dots. When plotting multiple 
        points, using an alpha value lower than one is recommended, to prevent
        outliers from distorting the shape of the distribution by chance. The 
        default is 0.4.
    rasterized : Bool or None, optional
        Whether to rasterize the resulting image. When exporting figures as 
        vector images, if the iamge has too many points, editing the resulting 
        file cna be hard. In such cases, it may be better to rasterize the 
        scatter plot and make sure to save it with a high dpi. If rasterized is
        None, then the plot will be rasterized when there are more than 100 
        points in the dataset. The default is None.
    width_distr : 'normal' or 'uniform', optional
        Decides the distribution from which to choose the horizontal offset of
        each point in the cloud. The default is 'normal'.
    orientation : 'horizontal' or 'vertical', optional
        Decides if the scatter plot extends horizontally or vertically, similar
        to the `vert` argument in matplotlib.pyplot.boxplot. Default is
        'vertical'.
    rng : np.random.Generator or None, optional
        A numpy random number generator to use. If None, a new default generator
        will be created. Use this to set the random seed for reproducibility. 
        The default is None.
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth. This can be 'scott',
        'silverman', a scalar constant or a callable. If a scalar, this will be 
        used directly as kde.factor. If a callable, it should take a gaussian_kde
        instance as only parameter and return a scalar. If None (default), 'scott'
        is used. See Notes in scipy.stats.gaussian_kde for more details.


    *plot_kw :
        Other keyword arguments to be passed to the plotting function.

    Returns
    -------
    None.

    """
    
    assert orientation in ('horizontal', 'vertical')

    series = np.asarray(series)
    y = series[~np.isnan(series)]

    kde = stats.gaussian_kde(y, bw_method) if len(y)>1 else (lambda y: np.ones_like(y, dtype=float))
    max_val = kde(y).max()
    
    # default axis
    if ax is None:
        ax = plt.gca()
    
    # rastrize image if there are too many points
    if rasterized is None:
        rasterized = series.size>100
    
    # get default values
    sampler = get_sampler(width_distr, rng)
    scale = choose_default_scale(width_distr, scale)
    category_jittered, series_value = sampler(loc, kde(y)/max_val * scale, size=len(y)), y
    
    # choose orientation
    if orientation == 'horizontal':
        horiz, vert = series_value, category_jittered
    else:
        horiz, vert = category_jittered, series_value

    # choose color plotting mode (color=... or c=...)
    scatter_kwargs = choose_color_plotting_mode(plot_kw, vert)
    
    # alias other plot kwargs to scatter-style
    scatter_kwargs = _alias_scatter_kwargs(scatter_kwargs)
    
    # draw with scatter 
    collection = ax.scatter(
        horiz, vert,
        alpha=alpha, rasterized=rasterized,
        **scatter_kwargs, # including color or c
    )
    return collection

    
def postaplot_columns(data, scale=None, ax=None, alpha=0.4, rasterized=None, orientation='vertical', **kw):
    """
    A set of measurements using the postaplot method. Data should be either 
    an itnerable where each element is a dataset, a dictionary or a pandas
    DataFrame. This function will repeatedly call postaplot for each element
    in the iterable, column in the dataframe or entry in the dictionary. In the 
    latter two cases the function will also rename the horizontal axis to 
    reflect the category names in the DataFrame or dictionary.

    See postaplot_engine for a description of the other parameters.
    """
    
    if isinstance(data, dict):
        values = data.values()
        names = list(data.keys())
    elif isinstance(data, pd.DataFrame):
        values = data.values.T
        names = data.columns
    else:
        values = data
        names = None
    
    if ax is None:
        ax = plt.gca()
    
    for i, x in enumerate(values):
        postaplot_engine(i, x, scale, ax, alpha, rasterized, orientation, **kw)
        
    if names is not None:
        positions = list(range(len(values)))
        ax.set_xticks(positions)
        ax.set_xticklabels(names)

def postaplot(
    data=None,
    *,
    x: Optional[Union[str, Sequence]] = None,
    y: Optional[Union[str, Sequence]] = None,
    hue: Optional[Union[str, Sequence]] = None,
    order: Optional[Sequence] = None,
    hue_order: Optional[Sequence] = None,
    dodge: Optional[Union[bool, float]] = False,
    orient: Optional[str] = None,
    color: Optional[Union[str, tuple]] = None,
    palette: Optional[Union[str, Mapping, Sequence]] = None,
    ax: Optional[Axes] = None,
    hue_norm: Optional[Union[tuple, Normalize]] = None,  # (vmin, vmax) for continuous hue; None = auto
    # above are standard seaborn-like parameters
    # extras specific to this plot type:
    horizontal_scale: Optional[float] = None,  # was `scale` in the engine
    width_distr: str = "normal",
    bw_method=None, # like in seaborn violinplot and other kde-based plots
    seed: Optional[int] = None,
    rasterized: Optional[bool] = None,
    alpha: float = 0.4, # it's here to set a default in a visible place
    hollow: bool = False,
    reference: Union[bool, Literal['legend','colorbar','auto']] = 'auto',  # True/'auto'/'legend'/'colorbar'/False
    **plot_kwa,
):
    """
    KDE-driven categorical scatter (Sina-style) with a Seaborn-like API.

    Parameters
    ----------
    data : DataFrame or None
        Long-form dataframe is preferred. If None, x/y/hue may be arrays.
    x, y : str or 1D array-like
        Categorical axis (str or array) and numeric axis (str or array).
        If `orient` is None, we infer orientation like seaborn: the variable
        that looks categorical becomes the categorical axis.
    hue : str or 1D array-like, optional
        Semantic grouping. When provided, colors are mapped by `palette`.
    order, hue_order : sequences, optional
        Explicit order of categories and hue levels.
    dodge : bool or float, default False
        If True and hue is given, dodge hue groups at each category. If float,
        use that as the total width to dodge across (default 0.6).
    orient : {"x","y"} or None
        Orientation (categorical on `x` or on `y`). If None, inferred.
    color : color-like, optional
        Single color if `hue` is None.
    palette : name, list, or dict, optional
        Color mapping for hue. If dict, keys are hue levels. If list, must
        match `hue_order` length. If name, resolved via seaborn/matplotlib.
    hue_norm : (vmin,vmax), Normalize or None
        For continuous hue, values to anchor the colormap or a Normalize instance.
        If None, use min/max of the data.
    ax : matplotlib Axes, optional
        Target axes (defaults to current axes).
    horizontal_scale : float, optional
        Lateral jitter scaling. If None, defaults depend on `width_distr`
        (0.1 for "normal", 0.4 for "uniform") to match engine behavior.
    width_distr : {"normal","uniform"}, default "normal"
        Stochastic law for lateral offsets.
    bw_method : see scipy.stats.gaussian_kde
        Bandwidth selection for KDE.
    seed : int, optional
        For reproducible jitter. Intrnally creates a np.random.Generator.
    rasterized : bool or None
        If None, rasterize only for large n (>=100 points in a subset).
    alpha : float, default 0.4
        Marker transparency.
    hollow : bool, default False
        If True, use hollow markers (edgecolor but no facecolor).
    reference : bool or {'legend','colorbar','auto'}, default 'auto'
        If 'auto', choose legend or colorbar based on hue type. If False,
        do not add either. If True, use 'auto'.
    **plot_kw :
        Passed to the underlying matplotlib plotting call (e.g., marker=".",
        markersize=..., linewidth=..., etc.).

    Returns
    -------
    ax : matplotlib Axes
    """
    # Resolve ax
    if ax is None:
        ax = plt.gca()

    # Resolve arrays from `data` if strings are provided
    x_vals = _resolve_variable(x, data)
    y_vals = _resolve_variable(y, data)
    hue_vals = _resolve_variable(hue, data)

    # Infer orientation like seaborn: one of x/y must be numeric, the other categorical
    orient = _infer_orient(x_vals, y_vals, orient)

    # Build categorical keys and numeric values based on orientation
    if orient == "x":
        cat = x_vals if x_vals is not None else np.zeros_like(y_vals, dtype=object)
        val = y_vals
        cat_axis = "x"
    else:
        cat = y_vals if y_vals is not None else np.zeros_like(x_vals, dtype=object)
        val = x_vals
        cat_axis = "y"
    orientation = ("vertical" if cat_axis == "x" else "horizontal")

    if val is None:
        raise ValueError("Numeric axis data is missing; provide `x` or `y` accordingly.")

    # Drop rows with missing numeric or cat or hue
    mask = np.isfinite(val)
    if cat is not None:
        mask &= pd.notna(cat)
    if hue_vals is not None:
        mask &= pd.notna(hue_vals)
    cat = np.asarray(cat)[mask]
    val = np.asarray(val)[mask]
    if hue_vals is not None:
        hue_vals = np.asarray(hue_vals)[mask]

    # Category order
    cats_unique = pd.unique(cat) if order is None else pd.Index(order)
    # Validate provided order includes only seen categories
    cats_unique = [c for c in cats_unique if np.any(cat == c)]

    # Hue levels & palette
    is_continuous_hue, hue_levels, pal_map, cmap, norm = _resolve_hue_mapping(
        hue_vals, hue_order, palette, hue_norm)

    # Category baseline integer positions
    cat_to_pos = {c: i for i, c in enumerate(cats_unique)}

    # Random generator
    rng = np.random.default_rng(seed)

    # Loop categories (and hues) → call engine
    drawn_any = False
    collections = []
    kde_kwa = dict(scale=horizontal_scale, ax=ax, alpha=alpha,
                rasterized=rasterized, hollow=hollow, width_distr=width_distr,
                orientation=orientation, rng=rng, bw_method=bw_method)
    for c in cats_unique:
        pos = cat_to_pos[c]
        m_cat = (cat == c)
        
        # if no hue, just draw once
        if hue_vals is None:
            # Single color for the whole cloud
            color_kw = {}
            if color is not None:
                color_kw["color"] = color
            
            # plot
            cloud = postaplot_engine(loc=pos, series=val[m_cat],
                                        **kde_kwa, **color_kw, **plot_kwa)

            # record
            if cloud is not None: 
                collections.append(cloud)
            drawn_any = True
            continue # to avoid nesting below

        if not dodge:
            # One cloud per category; per-point colors from hue
            if is_continuous_hue:
                colors = cmap(norm(hue_vals[m_cat])) # type: ignore (norm and cmap never None when is_continuous_hue)
            else:
                # discrete, per-point colors by level
                hv_cat = hue_vals[m_cat]
                # map each level to its color with a vectorized lookup
                colors = np.vectorize(lambda h: pal_map.get(h, color))(hv_cat)
            
            # plot with per-point colors
            cloud = postaplot_engine(loc=pos, series=val[m_cat],
                                        **kde_kwa, color=colors, **plot_kwa)
            if cloud is not None: collections.append(cloud)
            drawn_any = True

        else:
            # dodge=True → only sensible for discrete hue
            if is_continuous_hue:
                # draw as a single cloud with continuous colors (dodging a continuum makes no sense)
                colors = cmap(norm(hue_vals[m_cat])) # type: ignore (norm and cmap never None when is_continuous_hue)
                cloud = postaplot_engine(loc=pos, series=val[m_cat],
                                            **kde_kwa, color=colors, **plot_kwa)

                # record
                if cloud is not None: 
                    collections.append(cloud)
                drawn_any = True

            else:
                # discrete hue + dodge=True → draw one cloud per hue level, dodged
                offsets = _dodge_offsets(len(hue_levels), dodge) 
                for off, h in zip(offsets, hue_levels):
                    m = m_cat & (hue_vals == h)
                    if not np.any(m): continue
                    cloud = postaplot_engine(loc=pos + off, series=val[m],
                                                **kde_kwa, color=pal_map.get(h, color), **plot_kwa)
                    # record
                    if cloud is not None: 
                        collections.append(cloud)
                drawn_any = True

    if not drawn_any:
        return []

    # Set legend or colorbar if needed
    add_legend_or_colorbar(
        ax, hue, plot_kwa, reference, is_continuous_hue, hue_levels, pal_map, cmap, norm
    )

    # Ticks/labels
    if cat_axis == "x":
        ax.set_xticks(list(cat_to_pos.values()))
        ax.set_xticklabels([str(c) for c in cats_unique])
    else:
        ax.set_yticks(list(cat_to_pos.values()))
        ax.set_yticklabels([str(c) for c in cats_unique])

    ax.set_xlabel(x if isinstance(x, str) else "")
    ax.set_ylabel(y if isinstance(y, str) else "")
    return collections
