import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from postaplot import postaplot  # public API

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
