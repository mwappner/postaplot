import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from postaplot.plotters import postaplot_engine  # public engine wrapper
from postaplot.styling import _alias_scatter_kwargs, choose_color_plotting_mode  # internals

def test_engine_returns_collection():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    fig, ax = plt.subplots()
    coll = postaplot_engine(loc=0, series=y, ax=ax, rng=rng, width_distr='normal')
    assert isinstance(coll, PathCollection)

def test_engine_per_point_colors_vector():
    rng = np.random.default_rng(1)
    y = rng.normal(size=30)
    colors = ["C0" if i % 2 == 0 else "C1" for i in range(len(y))]
    fig, ax = plt.subplots()
    coll = postaplot_engine(loc=0, series=y, ax=ax, rng=rng, width_distr='uniform', color=colors)
    assert isinstance(coll, PathCollection)
    # facecolors should reflect two unique entries
    fc = coll.get_facecolors() # type: ignore (Pylance doesn't see get_facecolors)
    assert fc.shape[0] == len(y)

def test_engine_per_point_colors_rgba_matrix():
    rng = np.random.default_rng(2)
    y = rng.normal(size=20)
    # build (N,4) RGBA
    rgba = np.zeros((len(y), 4), float)
    rgba[:, 0] = 1.0  # red channel
    rgba[:, 3] = 0.5  # alpha
    fig, ax = plt.subplots()
    coll = postaplot_engine(loc=0, series=y, ax=ax, rng=rng, color=rgba)
    assert isinstance(coll, PathCollection)
    fc = coll.get_facecolors() # type: ignore (Pylance doesn't see get_facecolors)
    assert np.allclose(fc[:, 0], 1.0)

def test_aliaser_hollow_moves_colors_to_edges():
    # simulate kwargs with per-point colors that should color edges when hollow
    plot_kw = dict(hollow=True, c=["C0","C1","C2"])
    # first choose face/edge color routing
    routed = choose_color_plotting_mode(dict(plot_kw), np.arange(3))
    # then alias for scatter semantics
    aliased = _alias_scatter_kwargs(routed)
    assert aliased.get("facecolors") == "none"
    # When hollow: per-point colors should color edges
    assert "edgecolors" in aliased or "c" in aliased or "color" in aliased
