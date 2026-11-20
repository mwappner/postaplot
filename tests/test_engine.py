import numpy as np
import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import PathCollection

from postaplot.plotters import postaplot_engine  # public engine wrapper
from postaplot.styling import _alias_scatter_kwargs, choose_color_plotting_mode  # internals


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
@pytest.mark.parametrize("side, sign", [("negative", -1), ("positive", 1)])
@pytest.mark.parametrize("width_distr", ["normal", "uniform"])
@pytest.mark.parametrize("width_method", ["kde", "histogram"])
def test_engine_one_sided_cloud_folds_symmetric_offsets(
    orientation, side, sign, width_distr, width_method
):
    series = np.array([-2.0, -1.0, -0.25, 0.5, 1.5, 2.0])
    loc = 1.75
    common = dict(
        loc=loc, series=series, orientation=orientation,
        width_distr=width_distr, width_method=width_method,
    )
    fig, (ax_both, ax_side) = plt.subplots(1, 2)
    both = postaplot_engine(ax=ax_both, rng=np.random.default_rng(42), **common)
    half = postaplot_engine(
        ax=ax_side, rng=np.random.default_rng(42), side=side, **common
    )

    both_xy = both.get_offsets()
    half_xy = half.get_offsets()
    cat_index, observation_index = ((0, 1) if orientation == "vertical" else (1, 0))
    assert np.allclose(
        half_xy[:, cat_index] - loc,
        sign * np.abs(both_xy[:, cat_index] - loc),
    )
    assert np.array_equal(half_xy[:, observation_index], series)
    plt.close(fig)


def test_engine_omitted_side_matches_both():
    series = np.linspace(-1, 1, 10)
    fig, (ax_default, ax_both) = plt.subplots(1, 2)
    default = postaplot_engine(
        2, series, ax=ax_default, rng=np.random.default_rng(7)
    )
    both = postaplot_engine(
        2, series, ax=ax_both, rng=np.random.default_rng(7), side="both"
    )
    assert np.array_equal(default.get_offsets(), both.get_offsets())


def test_engine_rejects_invalid_side():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="side must be"):
        postaplot_engine(0, [1.0, 2.0], ax=ax, side="center")
    assert not ax.collections

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
