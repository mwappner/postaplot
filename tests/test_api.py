import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pytest

from postaplot import postaplot  # public API


def _split_df():
    return pd.DataFrame({
        "grp": ["B", "B", "B", "B", "A", "A", "A", "C", "C"],
        "value": [1., 2., 3., 4., 5., 6., 7., 8., 9.],
        "hue": ["left", "left", "right", "right", "left", "right", "right", "right", "right"],
    })


@pytest.mark.parametrize("orient", ["x", "y"])
def test_split_geometry_order_palette_and_legend(orient):
    df = _split_df()
    x, y = (("grp", "value") if orient == "x" else ("value", "grp"))
    palette = {"left": "#cc0000", "right": "#0000cc"}
    fig, ax = plt.subplots()
    clouds = postaplot(
        df, x=x, y=y, hue="hue", split=True, order=["A", "B"],
        palette=palette, seed=4, ax=ax,
    )

    assert len(clouds) == 4
    cat_index = 0 if orient == "x" else 1
    for cloud, pos, sign in zip(clouds, [0, 0, 1, 1], [-1, 1, -1, 1]):
        offsets = cloud.get_offsets()[:, cat_index] - pos
        assert np.all(sign * offsets >= 0)
    assert np.allclose(clouds[0].get_facecolors()[0], mcolors.to_rgba(palette["left"], .4))
    assert np.allclose(clouds[1].get_facecolors()[0], mcolors.to_rgba(palette["right"], .4))
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["left", "right"]
    ticklabels = ax.get_xticklabels() if orient == "x" else ax.get_yticklabels()
    assert [tick.get_text() for tick in ticklabels] == ["A", "B"]


def test_split_reversed_hue_order_swaps_sides_and_missing_level_keeps_assignment():
    df = _split_df()
    fig, ax = plt.subplots()
    clouds = postaplot(
        df, x="grp", y="value", hue="hue", split=True,
        hue_order=["right", "left"], order=["B", "C"], seed=5, ax=ax,
    )

    # B has both levels; C has only "right", which remains on the negative side.
    assert len(clouds) == 3
    assert np.all(clouds[0].get_offsets()[:, 0] <= 0)
    assert np.all(clouds[1].get_offsets()[:, 0] >= 0)
    assert np.all(clouds[2].get_offsets()[:, 0] <= 1)


def test_split_seeded_reproducibility():
    df = _split_df()
    figures = [plt.subplots() for _ in range(2)]
    results = [
        postaplot(df, x="grp", y="value", hue="hue", split=True, seed=9, ax=ax)
        for _, ax in figures
    ]
    for first, second in zip(*results):
        assert np.array_equal(first.get_offsets(), second.get_offsets())


@pytest.mark.parametrize("extra, match", [
    ({}, "discrete hue"),
    ({"hue": "one"}, "exactly two"),
    ({"hue": "many"}, "continuous hue"),
    ({"hue": "two", "dodge": True}, "dodge"),
    ({"hue": "two", "box": True}, "box=True"),
])
def test_invalid_split_draws_no_artists(extra, match):
    df = pd.DataFrame({
        "grp": ["A"] * 10,
        "value": np.arange(10, dtype=float),
        "one": ["only"] * 10,
        "two": ["u", "v"] * 5,
        "many": np.arange(10, dtype=float),
    })
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=match):
        postaplot(df, x="grp", y="value", split=True, ax=ax, **extra)
    assert not ax.collections
    assert not ax.lines
    assert not ax.patches


def test_omitted_split_matches_false():
    df = _split_df()
    fig, (ax_default, ax_false) = plt.subplots(1, 2)
    default = postaplot(df, x="grp", y="value", hue="hue", seed=11, ax=ax_default)
    explicit = postaplot(
        df, x="grp", y="value", hue="hue", split=False, seed=11, ax=ax_false
    )
    assert len(default) == len(explicit)
    for first, second in zip(default, explicit):
        assert np.array_equal(first.get_offsets(), second.get_offsets())

def _toy_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 150
    grp = np.repeat(["A","B","C"], n)
    y = np.r_[rng.normal(0,1,n), rng.normal(1,1,n), rng.standard_t(4, n)]
    return pd.DataFrame({"grp": grp, "y": y})

def test_basic_no_hue_returns_collections_and_sets_ticks_labels():
    df = _toy_df()
    fig, ax = plt.subplots()
    colls = postaplot(data=df, x="grp", y="y", ax=ax, seed=0)
    assert isinstance(colls, list) and len(colls) == 3  # one cloud per group
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A","B","C"]
    # labels from strings
    assert ax.get_xlabel() == "grp"
    assert ax.get_ylabel() == "y"

def test_discrete_hue_no_dodge_one_cloud_per_group():
    df = _toy_df()
    # build a balanced discrete hue
    df["h"] = np.tile(np.repeat(["u","v"], len(df)//6), 3)[:len(df)]
    fig, ax = plt.subplots()
    colls = postaplot(data=df, x="grp", y="y", hue="h", dodge=False, ax=ax, seed=1)
    assert len(colls) == 3  # still one cloud per category

def test_discrete_hue_with_dodge_subclouds():
    df = _toy_df()
    df["h"] = np.tile(np.repeat(["u","v"], len(df)//6), 3)[:len(df)]
    fig, ax = plt.subplots()
    colls = postaplot(data=df, x="grp", y="y", hue="h", dodge=True, ax=ax, seed=1)
    assert len(colls) == 3 * 2  # two hue levels per category

def test_continuous_hue_adds_colorbar_when_auto():
    df = _toy_df()
    # continuous hue (many unique numeric values)
    df["h"] = np.linspace(0, 1, len(df))
    fig, ax = plt.subplots()
    _ = postaplot(data=df, x="grp", y="y", hue="h", reference="auto", ax=ax, seed=2)
    # colorbar usually adds a second axes to the figure
    assert len(fig.axes) >= 2
