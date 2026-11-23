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


@pytest.mark.parametrize("alias, canonical", [("v", "x"), ("h", "y")])
@pytest.mark.parametrize("split_box", [False, True])
def test_orient_alias_matches_canonical(alias, canonical, split_box):
    df = pd.DataFrame({
        "category": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "value": [1., 2., 3., 4., 5., 6., 7., 8.],
        "hue": ["u", "u", "v", "v", "u", "u", "v", "v"],
    })
    x, y = (
        ("category", "value") if canonical == "x"
        else ("value", "category")
    )
    hue = "hue" if split_box else None
    fig, (ax_alias, ax_canonical) = plt.subplots(1, 2)
    alias_artists = postaplot(
        df, x=x, y=y, hue=hue, orient=alias,
        split=split_box, box=split_box, seed=28, ax=ax_alias,
    )
    canonical_artists = postaplot(
        df, x=x, y=y, hue=hue, orient=canonical,
        split=split_box, box=split_box, seed=28, ax=ax_canonical,
    )

    n_clouds = 4 if split_box else 2
    assert len(alias_artists) == len(canonical_artists)
    for alias_cloud, canonical_cloud in zip(
        alias_artists[:n_clouds], canonical_artists[:n_clouds]
    ):
        assert np.array_equal(
            alias_cloud.get_offsets(), canonical_cloud.get_offsets()
        )

    if split_box:
        for alias_box, canonical_box in zip(
            alias_artists[n_clouds:], canonical_artists[n_clouds:]
        ):
            assert np.array_equal(
                alias_box["boxes"][0].get_path().vertices,
                canonical_box["boxes"][0].get_path().vertices,
            )
            for key in ("medians", "whiskers", "caps"):
                for alias_line, canonical_line in zip(
                    alias_box[key], canonical_box[key]
                ):
                    assert np.array_equal(
                        alias_line.get_xydata(), canonical_line.get_xydata()
                    )

    alias_ticks = (
        ax_alias.get_xticklabels() if canonical == "x"
        else ax_alias.get_yticklabels()
    )
    canonical_ticks = (
        ax_canonical.get_xticklabels() if canonical == "x"
        else ax_canonical.get_yticklabels()
    )
    assert [tick.get_text() for tick in alias_ticks] == ["A", "B"]
    assert [tick.get_text() for tick in canonical_ticks] == ["A", "B"]


def test_invalid_orient_draws_no_artists():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="orient must be"):
        postaplot(
            x=["A", "A"], y=[1., 2.], orient="diagonal", ax=ax
        )
    assert not ax.collections
    assert not ax.lines
    assert not ax.patches


@pytest.mark.parametrize("orient", ["x", "y"])
@pytest.mark.parametrize("side, sign", [("negative", -1), ("positive", 1)])
@pytest.mark.parametrize("with_hue", [False, True])
def test_single_half_cloud_geometry_color_and_legend(orient, side, sign, with_hue):
    df = pd.DataFrame({
        "grp": ["A"] * 4,
        "value": [1., 2., 3., 4.],
        "hue": ["only"] * 4,
    })
    x, y = (("grp", "value") if orient == "x" else ("value", "grp"))
    hue = "hue" if with_hue else None
    palette = {"only": "#336699"} if with_hue else None
    fig, ax = plt.subplots()
    clouds = postaplot(
        df, x=x, y=y, hue=hue, split=True, split_side=side,
        color=None if with_hue else "#993366", palette=palette,
        seed=2, ax=ax,
    )

    assert len(clouds) == 1
    categorical_index, value_index = ((0, 1) if orient == "x" else (1, 0))
    offsets = clouds[0].get_offsets()
    assert np.all(sign * offsets[:, categorical_index] >= 0)
    assert np.array_equal(offsets[:, value_index], df["value"])
    expected_color = palette["only"] if with_hue else "#993366"
    assert np.allclose(
        clouds[0].get_facecolors()[0], mcolors.to_rgba(expected_color, .4)
    )
    if with_hue:
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ["only"]
    else:
        assert ax.get_legend() is None


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


def test_split_side_mirrors_two_hues_without_reordering_them():
    df = _split_df()
    fig, (ax_negative, ax_positive) = plt.subplots(1, 2)
    negative = postaplot(
        df, x="grp", y="value", hue="hue", split=True,
        split_side="negative", seed=10, ax=ax_negative,
    )
    positive = postaplot(
        df, x="grp", y="value", hue="hue", split=True,
        split_side="positive", seed=10, ax=ax_positive,
    )

    assert len(negative) == len(positive)
    category_positions = [0, 0, 1, 1, 2]
    for loc, first, mirrored in zip(category_positions, negative, positive):
        assert np.allclose(
            mirrored.get_offsets()[:, 0] - loc,
            -(first.get_offsets()[:, 0] - loc),
        )
        assert np.array_equal(
            mirrored.get_offsets()[:, 1], first.get_offsets()[:, 1]
        )
        assert np.allclose(mirrored.get_facecolors(), first.get_facecolors())
    expected_legend = ["left", "right"]
    assert [
        t.get_text() for t in ax_negative.get_legend().get_texts()
    ] == expected_legend
    assert [
        t.get_text() for t in ax_positive.get_legend().get_texts()
    ] == expected_legend


@pytest.mark.parametrize("extra, match", [
    ({"hue": "many"}, "continuous hue"),
])
def test_invalid_split_draws_no_artists(extra, match):
    df = pd.DataFrame({
        "grp": ["A"] * 10,
        "value": np.arange(10, dtype=float),
        "one": ["only"] * 10,
        "two": ["u", "v"] * 5,
        "three": ["u", "v", "w", "u", "v"] * 2,
        "many": np.arange(10, dtype=float),
    })
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=match):
        postaplot(df, x="grp", y="value", split=True, ax=ax, **extra)
    assert not ax.collections
    assert not ax.lines
    assert not ax.patches


@pytest.mark.parametrize("n_hues", [3, 4])
@pytest.mark.parametrize("orient", ["x", "y"])
@pytest.mark.parametrize("side", ["negative", "positive"])
def test_multi_hue_split_slots_directions_and_legend(n_hues, orient, side):
    levels = [f"h{i}" for i in range(n_hues)]
    df = pd.DataFrame({
        "grp": ["A"] * (2 * n_hues),
        "value": np.ravel([[i, i + .5] for i in range(n_hues)]),
        "hue": np.repeat(levels, 2),
    })
    x, y = (("grp", "value") if orient == "x" else ("value", "grp"))
    fig, ax = plt.subplots()
    clouds = postaplot(
        df, x=x, y=y, hue="hue", split=True, split_side=side,
        width_distr="uniform", width_method="histogram", seed=18, ax=ax,
    )

    edges = np.linspace(-.2, .2, n_hues + 1)
    categorical_index, value_index = ((0, 1) if orient == "x" else (1, 0))
    for i, cloud in enumerate(clouds):
        expected_side = side if i % 2 == 0 else (
            "positive" if side == "negative" else "negative"
        )
        anchor = edges[i + 1] if expected_side == "negative" else edges[i]
        categorical = cloud.get_offsets()[:, categorical_index]
        assert np.all(categorical >= edges[i])
        assert np.all(categorical <= edges[i + 1])
        if expected_side == "negative":
            assert np.all(categorical <= anchor)
        else:
            assert np.all(categorical >= anchor)
        assert np.array_equal(
            cloud.get_offsets()[:, value_index], [i, i + .5]
        )
    assert [text.get_text() for text in ax.get_legend().get_texts()] == levels


def test_multi_hue_order_and_missing_level_keep_fixed_slots():
    hue_order = ["d", "b", "a", "c"]
    df = pd.DataFrame({
        "grp": np.repeat(["A", "B"], [8, 6]),
        "value": np.arange(14, dtype=float),
        "hue": ["a", "a", "b", "b", "c", "c", "d", "d",
                "a", "a", "c", "c", "d", "d"],
    })
    fig, ax = plt.subplots()
    clouds = postaplot(
        df, x="grp", y="value", hue="hue", hue_order=hue_order,
        split=True, width_distr="uniform", width_method="histogram",
        seed=19, ax=ax,
    )

    assert len(clouds) == 7
    edges = np.linspace(-.2, .2, 5)
    # Category B has no "b", so its second slot stays empty while a/c retain
    # the third/fourth slots assigned by the global hue order.
    for cloud, slot in zip(clouds[4:], [0, 2, 3]):
        categorical = cloud.get_offsets()[:, 0] - 1
        assert np.all(categorical >= edges[slot])
        assert np.all(categorical <= edges[slot + 1])
    assert [text.get_text() for text in ax.get_legend().get_texts()] == hue_order


@pytest.mark.parametrize("n_hues", [3, 4])
def test_multi_hue_split_boxes_follow_slot_anchors(n_hues):
    levels = [f"h{i}" for i in range(n_hues)]
    df = pd.DataFrame({
        "grp": ["A"] * (4 * n_hues),
        "value": np.ravel([[i, i + .25, i + .5, i + .75]
                           for i in range(n_hues)]),
        "hue": np.repeat(levels, 4),
    })
    fig, ax = plt.subplots()
    artists = postaplot(
        df, x="grp", y="value", hue="hue", split=True, box=True,
        width_distr="uniform", width_method="histogram", seed=20, ax=ax,
    )

    boxes = artists[n_hues:]
    edges = np.linspace(-.2, .2, n_hues + 1)
    scale_factor = 1 / n_hues
    expected_width = .24 * scale_factor
    expected_inset = (.24 + .08) * scale_factor / 2
    for i, box in enumerate(boxes):
        side = "negative" if i % 2 == 0 else "positive"
        sign = -1 if side == "negative" else 1
        anchor = edges[i + 1] if side == "negative" else edges[i]
        center = np.mean(box["medians"][0].get_xdata())
        assert np.isclose(center, anchor + sign * expected_inset)
        box_x = box["boxes"][0].get_path().vertices[:, 0]
        assert np.isclose(np.max(box_x) - np.min(box_x), expected_width)
        inner_edge = np.max(box_x) if side == "negative" else np.min(box_x)
        for whisker in box["whiskers"]:
            assert np.allclose(whisker.get_xdata(), inner_edge)
        for cap in box["caps"]:
            cap_x = cap.get_xdata()
            assert np.isclose(cap_x[0], inner_edge)
            assert sign * (cap_x[1] - inner_edge) > 0


@pytest.mark.parametrize("split", [False, True])
def test_invalid_split_side_draws_no_artists(split):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="split_side must be"):
        postaplot(
            x=["A", "A"], y=[1., 2.], split=split,
            split_side="center", ax=ax,
        )
    assert not ax.collections
    assert not ax.lines
    assert not ax.patches


def test_split_side_has_no_effect_when_split_is_false():
    df = _split_df()
    fig, (ax_negative, ax_positive) = plt.subplots(1, 2)
    negative = postaplot(
        df, x="grp", y="value", hue="hue", split=False,
        split_side="negative", seed=11, ax=ax_negative,
    )
    positive = postaplot(
        df, x="grp", y="value", hue="hue", split=False,
        split_side="positive", seed=11, ax=ax_positive,
    )
    for first, second in zip(negative, positive):
        assert np.array_equal(first.get_offsets(), second.get_offsets())


@pytest.mark.parametrize("orient", ["x", "y"])
@pytest.mark.parametrize("side, sign", [("negative", -1), ("positive", 1)])
@pytest.mark.parametrize("with_hue", [False, True])
def test_single_half_box_follows_selected_side(orient, side, sign, with_hue):
    df = pd.DataFrame({
        "grp": ["A"] * 4,
        "value": [1., 2., 3., 4.],
        "hue": ["only"] * 4,
    })
    x, y = (("grp", "value") if orient == "x" else ("value", "grp"))
    fig, ax = plt.subplots()
    artists = postaplot(
        df, x=x, y=y, hue="hue" if with_hue else None,
        split=True, split_side=side, box=True, seed=12, ax=ax,
    )

    cloud, box = artists
    categorical_index = 0 if orient == "x" else 1
    assert np.all(sign * cloud.get_offsets()[:, categorical_index] >= 0)
    category_coordinate = (
        lambda artist: artist.get_xdata()
        if orient == "x" else artist.get_ydata()
    )
    center = np.mean(category_coordinate(box["medians"][0]))
    assert np.isclose(center, sign * .08)
    box_coordinates = box["boxes"][0].get_path().vertices[:, categorical_index]
    inner_edge = np.max(box_coordinates) if sign < 0 else np.min(box_coordinates)
    for whisker in box["whiskers"]:
        assert np.allclose(category_coordinate(whisker), inner_edge)
    for cap in box["caps"]:
        cap_coordinates = category_coordinate(cap)
        assert np.isclose(cap_coordinates[0], inner_edge)
        assert sign * (cap_coordinates[1] - inner_edge) > 0


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
    for edge in box_edges:
        assert np.isclose(np.max(edge) - np.min(edge), .12)

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


@pytest.mark.parametrize("orient", ["x", "y"])
@pytest.mark.parametrize("gap", [0, .25])
@pytest.mark.parametrize("explicit_width", [False, True])
def test_split_box_width_and_anchor_clearance(
    orient, gap, explicit_width
):
    df = _split_df().query("grp == 'B'")
    x, y = (("grp", "value") if orient == "x" else ("value", "grp"))
    box_kwa = {"widths": .1} if explicit_width else {}
    fig, ax = plt.subplots()
    artists = postaplot(
        df, x=x, y=y, hue="hue", split=True, box=True,
        gap=gap, box_kwa=box_kwa, seed=27, ax=ax,
    )

    boxes = artists[2:]
    category_index = 0 if orient == "x" else 1
    category_coordinate = (
        lambda artist: artist.get_xdata()
        if orient == "x" else artist.get_ydata()
    )
    factor = 1 - gap
    effective_width = (.1 if explicit_width else .12) * factor
    clearance = .02 * factor
    anchors = [-.1 * gap, .1 * gap]
    for sign, anchor, box in zip([-1, 1], anchors, boxes):
        box_axis = box["boxes"][0].get_path().vertices[:, category_index]
        assert np.isclose(np.max(box_axis) - np.min(box_axis), effective_width)
        inner_edge = np.max(box_axis) if sign < 0 else np.min(box_axis)
        assert np.isclose(abs(inner_edge - anchor), clearance)
        center = np.mean(category_coordinate(box["medians"][0]))
        assert np.isclose(
            center, anchor + sign * (effective_width / 2 + clearance)
        )


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
        box_x = box["boxes"][0].get_path().vertices[:, 0]
        assert np.isclose(np.max(box_x) - np.min(box_x), .4)
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
    for center, box in zip([-.1, .1], boxes):
        for whisker in box["whiskers"]:
            assert np.allclose(whisker.get_xdata(), center)
        box_x = box["boxes"][0].get_path().vertices[:, 0]
        assert np.isclose(np.max(box_x) - np.min(box_x), .1)


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


@pytest.mark.parametrize("orient", ["x", "y"])
@pytest.mark.parametrize("width_distr", ["normal", "uniform"])
def test_width_scales_unsplit_jitter_and_preserves_values(orient, width_distr):
    values = np.array([1., 2., 3., 4.])
    x, y = ((["A"] * 4, values) if orient == "x" else (values, ["A"] * 4))
    fig, (ax_narrow, ax_wide) = plt.subplots(1, 2)
    narrow = postaplot(
        x=x, y=y, orient=orient, width=.4, width_distr=width_distr,
        width_method="histogram", seed=21, ax=ax_narrow,
    )[0]
    wide = postaplot(
        x=x, y=y, orient=orient, width=.8, width_distr=width_distr,
        width_method="histogram", seed=21, ax=ax_wide,
    )[0]

    categorical_index, value_index = ((0, 1) if orient == "x" else (1, 0))
    assert np.allclose(
        wide.get_offsets()[:, categorical_index],
        2 * narrow.get_offsets()[:, categorical_index],
    )
    assert np.array_equal(wide.get_offsets()[:, value_index], values)


@pytest.mark.parametrize("orient", ["x", "y"])
def test_dodge_width_gap_slots_and_missing_level(orient):
    df = pd.DataFrame({
        "grp": ["A"] * 6 + ["B"] * 4,
        "value": np.arange(10, dtype=float),
        "hue": ["u", "u", "v", "v", "w", "w",
                "u", "u", "w", "w"],
    })
    x, y = (("grp", "value") if orient == "x" else ("value", "grp"))
    fig, ax = plt.subplots()
    clouds = postaplot(
        df, x=x, y=y, hue="hue", dodge=True, width=.6, gap=.25,
        width_distr="uniform", width_method="histogram", seed=22, ax=ax,
    )

    categorical_index = 0 if orient == "x" else 1
    centers = [-.2, 0., .2, .8, 1.2]
    assert len(clouds) == len(centers)
    for cloud, center in zip(clouds, centers):
        offsets = cloud.get_offsets()[:, categorical_index]
        assert np.all(offsets >= center - .075)
        assert np.all(offsets <= center + .075)


@pytest.mark.parametrize("width_distr", ["normal", "uniform"])
def test_dodge_gap_keeps_slot_centers_and_scales_jitter(width_distr):
    df = pd.DataFrame({
        "grp": ["A"] * 8,
        "value": np.arange(8, dtype=float),
        "hue": ["u"] * 4 + ["v"] * 4,
    })
    fig, (ax_full, ax_gap) = plt.subplots(1, 2)
    common = dict(
        data=df, x="grp", y="value", hue="hue", dodge=True,
        width=.8, width_distr=width_distr,
        width_method="histogram", seed=26,
    )
    full = postaplot(gap=0, ax=ax_full, **common)
    narrowed = postaplot(gap=.5, ax=ax_gap, **common)
    for center, first, second in zip([-.2, .2], full, narrowed):
        assert np.allclose(
            second.get_offsets()[:, 0] - center,
            .5 * (first.get_offsets()[:, 0] - center),
        )
        assert np.array_equal(
            second.get_offsets()[:, 1], first.get_offsets()[:, 1]
        )


def test_split_gap_insets_anchors_and_redundant_dodge():
    levels = ["a", "b", "c", "d"]
    df = pd.DataFrame({
        "grp": ["A"] * 8,
        "value": np.arange(8, dtype=float),
        "hue": np.repeat(levels, 2),
    })
    fig, (ax_plain, ax_dodge) = plt.subplots(1, 2)
    common = dict(
        data=df, x="grp", y="value", hue="hue", split=True,
        width=.8, gap=.25, width_distr="uniform",
        width_method="histogram", seed=23,
    )
    plain = postaplot(ax=ax_plain, **common)
    redundant = postaplot(ax=ax_dodge, dodge=True, **common)

    usable_ranges = [(-.375, -.225), (-.175, -.025),
                     (.025, .175), (.225, .375)]
    for first, second, (lower, upper) in zip(plain, redundant, usable_ranges):
        assert np.array_equal(first.get_offsets(), second.get_offsets())
        categorical = first.get_offsets()[:, 0]
        assert np.all(categorical >= lower)
        assert np.all(categorical <= upper)


def test_split_gap_scales_box_and_cap_geometry():
    fig, ax = plt.subplots()
    artists = postaplot(
        x=["A"] * 5, y=[0., 1., 2., 3., 4.], split=True,
        split_side="negative", box=True, width=.8, gap=.5,
        box_kwa={"widths": .1, "capwidths": .08}, seed=24, ax=ax,
    )
    box = artists[1]
    box_x = box["boxes"][0].get_path().vertices[:, 0]
    assert np.isclose(np.max(box_x) - np.min(box_x), .05)
    assert np.isclose(np.mean(box["medians"][0].get_xdata()), -.145)
    inner_edge = np.max(box_x)
    assert np.isclose(inner_edge, -.12)
    for whisker in box["whiskers"]:
        assert np.allclose(whisker.get_xdata(), inner_edge)
    for cap in box["caps"]:
        assert np.isclose(cap.get_xdata()[0], inner_edge)
        assert np.isclose(cap.get_xdata()[1], inner_edge - .02)


@pytest.mark.parametrize("width_distr, legacy, replacement", [
    ("normal", .1, .4),
    ("uniform", .2, .2),
])
def test_horizontal_scale_warns_and_suggested_width_matches(
    width_distr, legacy, replacement
):
    values = [1., 2., 3., 4.]
    fig, (ax_legacy, ax_width) = plt.subplots(1, 2)
    with pytest.warns(FutureWarning, match=fr"width={replacement:g}"):
        old = postaplot(
            x=["A"] * 4, y=values, horizontal_scale=legacy,
            width_distr=width_distr, seed=25, ax=ax_legacy,
        )[0]
    new = postaplot(
        x=["A"] * 4, y=values, width=replacement,
        width_distr=width_distr, seed=25, ax=ax_width,
    )[0]
    assert np.array_equal(old.get_offsets(), new.get_offsets())


@pytest.mark.parametrize("kwargs, match", [
    ({"width": 0}, "positive finite"),
    ({"width": np.inf}, "positive finite"),
    ({"gap": -0.1}, r"in \[0, 1\)"),
    ({"gap": 1}, r"in \[0, 1\)"),
    ({"dodge": .3}, "dodge must be a boolean"),
    ({"width": .8, "horizontal_scale": .1}, "cannot be specified together"),
])
def test_invalid_width_dodge_gap_draws_no_artists(kwargs, match):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=match):
        postaplot(x=["A", "A"], y=[1., 2.], ax=ax, **kwargs)
    assert not ax.collections
    assert not ax.lines
    assert not ax.patches

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
