"""
Generate figures for docs/cookbook.md.
Run from the project root (the folder containing `postaplot/` and `docs/`):

    python scripts/generate_figures.py

Images are saved under docs/figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import CenteredNorm

from postaplot import postaplot, postaplot_columns, postaplot_engine


OUT = os.path.join("docs", "figures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(12)
data = pd.concat(
    [
        pd.DataFrame({
            "stage": stage,
            "condition": condition,
            "response": rng.normal(mean + shift, spread, 120),
        })
        for stage, mean in [("Early", 0), ("Late", 1.2), ("Middle", 0.6)]
        for condition, shift, spread in [
            ("Control", 0, 0.45),
            ("Treated", 0.35, 0.70),
        ]
    ],
    ignore_index=True,
)
data["score"] = data["response"] + rng.normal(0, 0.35, len(data))

stage_order = ["Early", "Middle", "Late"]
condition_order = ["Treated", "Control"]
palette = {"Control": "goldenrod", "Treated": "darkseagreen"}


def savefig(fig, filename):
    path = os.path.join(OUT, filename)
    fig.savefig(path, dpi=150)
    print("Wrote", path)
    plt.close(fig)


def basic_example():
    fig, ax = plt.subplots()
    postaplot(data, x="stage", y="response", seed=7, ax=ax)
    savefig(fig, "basic_example.png")


def ordering():
    fig, ax = plt.subplots()
    postaplot(
        data, x="stage", y="response", hue="condition",
        order=stage_order, hue_order=condition_order, palette=palette,
        dodge=True, width=0.7, gap=0.15, seed=7, ax=ax,
    )
    savefig(fig, "ordering.png")


def dodge_and_split():
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4), sharey=True, layout="constrained"
    )
    for ax, layout, title in [
        (axes[0], {}, "Overlapping"),
        (axes[1], {"dodge": True}, "Dodged"),
        (axes[2], {"split": True}, "Split"),
    ]:
        postaplot(
            data, x="stage", y="response", hue="condition",
            order=stage_order, hue_order=condition_order, palette=palette,
            width=0.7, seed=7, ax=ax, **layout,
        )
        ax.set_title(title)
    savefig(fig, "dodge_and_split.png")


def horizontal_half_cloud():
    fig, ax = plt.subplots()
    postaplot(
        data, x="response", y="stage", order=stage_order,
        orient="h", split=True, split_side="positive", seed=7, ax=ax,
    )
    savefig(fig, "horizontal_half_cloud.png")


def width_and_gap():
    fig, axes = plt.subplots(
        1, 3, figsize=(11, 3.5), sharey=True, layout="constrained"
    )
    for ax, width, gap in zip(axes, [0.4, 0.8, 0.8], [0, 0, 0.3]):
        postaplot(
            data, x="stage", y="response", hue="condition",
            order=stage_order, hue_order=condition_order, palette=palette,
            dodge=True, width=width, gap=gap, reference=ax is axes[0],
            seed=7, ax=ax,
        )
        ax.set_title(f"width={width}, gap={gap}")
    savefig(fig, "width_and_gap.png")


def split_box_styles():
    fig, axes = plt.subplots(
        1, 2, figsize=(9, 4), sharey=True, layout="constrained"
    )
    for ax, options, title in [
        (axes[0], {}, "Default boxes"),
        (axes[1], {
            "marker": "D", "markersize": 3, "alpha": 0.55, "hollow": True,
            "box_kwa": {
                "boxprops": {"facecolor": "whitesmoke", "edgecolor": "0.3"},
                "medianprops": {"color": "red", "linewidth": 2},
                "showfliers": False,
            },
        }, "Custom boxes and markers"),
    ]:
        postaplot(
            data, x="stage", y="response", hue="condition",
            order=stage_order, hue_order=condition_order, palette=palette,
            split=True, width=0.7, gap=0.15, box=True, seed=7, ax=ax,
            **options,
        )
        ax.set_title(title)
    savefig(fig, "split_box_styles.png")


def cloud_shape():
    fig, axes = plt.subplots(
        1, 3, figsize=(10, 3.5), sharey=True, layout="constrained"
    )
    for ax, distribution, method, title in [
        (axes[0], "normal", "kde", "Normal · KDE"),
        (axes[1], "uniform", "kde", "Uniform · KDE"),
        (axes[2], "uniform", "histogram", "Uniform · histogram"),
    ]:
        postaplot(
            data, x="stage", y="response", order=stage_order,
            width=0.65, width_distr=distribution, width_method=method,
            seed=7, ax=ax,
        )
        ax.set_title(title)
    savefig(fig, "cloud_shape.png")


def continuous_hue():
    fig, axes = plt.subplots(
        1, 2, figsize=(9, 4), sharey=True, layout="constrained"
    )
    for ax, reference in zip(axes, [True, False]):
        postaplot(
            data, x="stage", y="response", hue="score", order=stage_order,
            palette="coolwarm", hue_norm=CenteredNorm(vcenter=0),
            hollow=True, markersize=4, reference=reference, seed=7, ax=ax,
        )
        ax.set_title("With colorbar" if reference else "Without colorbar")
    savefig(fig, "continuous_hue.png")


def columns():
    wide = data.pivot(columns="stage", values="response").loc[:, stage_order]
    fig, ax = plt.subplots()
    postaplot_columns(wide, ax=ax, rng=np.random.default_rng(7))
    savefig(fig, "columns.png")


def face_and_edge_colors():
    middle = data.loc[data["stage"] == "Middle"]
    cmap = plt.get_cmap("coolwarm")
    norm = CenteredNorm(vcenter=0)
    facecolors = cmap(norm(middle["score"].to_numpy()))
    edgecolors = middle["condition"].map(palette).to_numpy()

    fig, ax = plt.subplots()
    cloud = postaplot_engine(
        loc=2, series=middle["response"], ax=ax,
        facecolors=facecolors, edgecolors=edgecolors,
        rng=np.random.default_rng(7),
        alpha=0.9, markersize=10, markeredgewidth=2
    )
    ax.set(xticks=[2], xticklabels=["Middle"], ylabel="response")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, label="score"
    )
    ax.legend(handles=[
        plt.Line2D(
            [], [], marker="o", linestyle="none", color="0.9",
            markeredgecolor=palette[level], markeredgewidth=2,
            label=level,
        )
        for level in condition_order
    ], title="condition")
    savefig(fig, "face_and_edge_colors.png")


def hue_markers():
    middle = data.loc[data["stage"] == "Middle"]
    fig, ax = plt.subplots()
    markers = {"Treated": "D", "Control": "o"}
    jitter_rng = np.random.default_rng(7)

    for condition in condition_order:
        values = middle.loc[middle["condition"] == condition, "response"]
        postaplot_engine(
            loc=0, series=values, ax=ax, rng=jitter_rng,
            color=palette[condition], marker=markers[condition],
            markersize=10, label=condition, alpha=0.9, markeredgecolor="none",
        )

    ax.set(xticks=[0], xticklabels=["Middle"], ylabel="response")
    ax.legend(title="condition")
    savefig(fig, "hue_markers.png")


def main():
    basic_example()
    ordering()
    dodge_and_split()
    horizontal_half_cloud()
    width_and_gap()
    split_box_styles()
    cloud_shape()
    continuous_hue()
    columns()
    face_and_edge_colors()
    hue_markers()


if __name__ == "__main__":
    main()
