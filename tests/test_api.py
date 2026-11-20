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


@pytest.mark.parametrize("orient", ["x", "y"])
def test_split_boxes_match_subsets_and_leave_center_gap(orient):
    df = _split_df().query("grp == 'B'")
    x, y = (("grp", "value") if orient == "x" else ("value", "grp"))
    fig, ax = plt.subplots()
    artists = postaplot(
        df, x=x, y=y, hue="hue", split=True, box=True,
        box_kwa={"capwidths": .08}, seed=12, ax=ax,
    )

    clouds, boxes = artists[:2], artists[2:]
    assert len(clouds) == len(boxes) == 2
    category_coordinate = (
        lambda artist: artist.get_xdata()
        if orient == "x" else artist.get_ydata()
    )
    centers = [np.mean(category_coordinate(box["medians"][0])) for box in boxes]
    assert centers[0] < 0 < centers[1]
    category_index = 0 if orient == "x" else 1
    box_edges = [
        box["boxes"][0].get_path().vertices[:, category_index]
        for box in boxes
    ]
    assert np.max(box_edges[0]) < 0 < np.min(box_edges[1])
    assert np.isclose(np.min(box_edges[1]) - np.max(box_edges[0]), .04)

    value_coordinate = (
        lambda artist: artist.get_ydata()
        if orient == "x" else artist.get_xdata()
    )
    assert np.isclose(np.mean(value_coordinate(boxes[0]["medians"][0])), 1.5)
    assert np.isclose(np.mean(value_coordinate(boxes[1]["medians"][0])), 3.5)
    expected_whisker_values = [([1.25, 1.], [1.75, 2.]),
                               ([3.25, 3.], [3.75, 4.])]
    for sign, center, inner_edge, box, expected_values in zip(
        [-1, 1], centers,
        [np.max(box_edges[0]), np.min(box_edges[1])],
        boxes, expected_whisker_values,
    ):
        for whisker in box["whiskers"]:
            assert np.allclose(category_coordinate(whisker), inner_edge)
        for cap in box["caps"]:
            cap_coordinates = category_coordinate(cap)
            assert np.isclose(cap_coordinates[0], inner_edge)
            assert np.isclose(cap_coordinates[1], inner_edge + sign * .04)
        for whisker, expected in zip(box["whiskers"], expected_values):
            assert np.allclose(value_coordinate(whisker), expected)
        for cap, endpoint in zip(box["caps"], [expected_values[0][1], expected_values[1][1]]):
            assert np.allclose(value_coordinate(cap), endpoint)


def test_split_box_hue_order_missing_hue_and_width_override():
    df = _split_df()
    fig, ax = plt.subplots()
    artists = postaplot(
        df, x="grp", y="value", hue="hue", split=True, box=True,
        hue_order=["right", "left"], order=["B", "C"],
        box_kwa={"widths": .2}, seed=13, ax=ax,
    )

    boxes = artists[3:]
    assert len(boxes) == 3
    centers = [np.mean(box["medians"][0].get_xdata()) for box in boxes]
    assert np.allclose(centers, [-.12, .12, .88])
    assert np.isclose(
        np.max(boxes[0]["boxes"][0].get_path().vertices[:, 0]), -.02
    )
    assert np.isclose(
        np.min(boxes[1]["boxes"][0].get_path().vertices[:, 0]), .02
    )
    assert not any(center > 1 for center in centers)
    inner_edges = [-.02, .02, .98]
    for inner_edge, box in zip(inner_edges, boxes):
        for whisker in box["whiskers"]:
            assert np.allclose(whisker.get_xdata(), inner_edge)


def test_split_box_fliers_stay_centered_and_caps_can_be_hidden():
    df = pd.DataFrame({
        "grp": ["A"] * 10,
        "value": [0., 1., 2., 3., 100.] * 2,
        "hue": ["left"] * 5 + ["right"] * 5,
    })
    fig, ax = plt.subplots()
    artists = postaplot(
        df, x="grp", y="value", hue="hue", split=True, box=True,
        box_kwa={"showfliers": True, "showcaps": False}, seed=16, ax=ax,
    )

    boxes = artists[2:]
    assert all(not box["caps"] for box in boxes)
    for center, box in zip([-.08, .08], boxes):
        assert np.allclose(box["fliers"][0].get_xdata(), center)
        assert np.array_equal(box["fliers"][0].get_ydata(), [100.])
        assert np.isclose(np.mean(box["medians"][0].get_xdata()), center)


def test_split_box_does_not_change_seeded_clouds():
    df = _split_df()
    fig, (ax_without, ax_with) = plt.subplots(1, 2)
    without = postaplot(
        df, x="grp", y="value", hue="hue", split=True,
        box=False, seed=14, ax=ax_without,
    )
    with_boxes = postaplot(
        df, x="grp", y="value", hue="hue", split=True,
        box=True, seed=14, ax=ax_with,
    )
    for cloud_without, cloud_with in zip(without, with_boxes[:len(without)]):
        assert np.array_equal(cloud_without.get_offsets(), cloud_with.get_offsets())


def test_non_split_box_behavior_is_preserved():
    df = _split_df()
    fig, ax = plt.subplots()
    artists = postaplot(
        df, x="grp", y="value", box=True, seed=15, ax=ax
    )
    assert len(artists) == 6
    for pos, box in enumerate(artists[3:]):
        assert np.isclose(np.mean(box["medians"][0].get_xdata()), pos)
        for whisker in box["whiskers"]:
            assert np.allclose(whisker.get_xdata(), pos)


def test_dodged_box_whiskers_remain_centered():
    df = _split_df()
    fig, ax = plt.subplots()
    artists = postaplot(
        df, x="grp", y="value", hue="hue", dodge=True, box=True,
        order=["B"], seed=17, ax=ax,
    )
    boxes = artists[2:]
    for center, box in zip([-.15, .15], boxes):
        for whisker in box["whiskers"]:
            assert np.allclose(whisker.get_xdata(), center)


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
