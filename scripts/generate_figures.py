"""
Generate example figures for the README and docs/cookbook.
Run from the project root (the folder containing `postaplot/` and `docs/`):

    python scripts/generate_figures.py

Images are saved under docs/figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib import colors

import postaplot

OUT = os.path.join("docs", "figures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(1)

def toy_df(n=100, seed=1, con_hue=False, dis_hue=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "group": np.repeat(["A","B","C"], n),
        "y": np.r_[rng.normal(0,1,n),
                   rng.normal(1,1,n),
                   rng.standard_t(4, n)]
    })
    if con_hue:
        df["hue_num"] = np.linspace(0, 1, len(df))
    if dis_hue:
        df["hue"] = np.tile(["u","v"], len(df)//2)
    return df

def savefig(ax, fname):
    path = os.path.join(OUT, fname)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    print("Wrote", path)

def basic_example():
    df = toy_df(100, seed=2)
    fig, ax = plt.subplots(figsize=(5,3.2))
    postaplot.postaplot(df, x="group", y="y", ax=ax, seed=0)
    savefig(ax, "basic_example.png"); plt.close(fig)

def jitter_normal_vs_uniform():
    df = toy_df(100, seed=3)
    fig, axs = plt.subplots(1, 2, figsize=(8,3.2), sharey=True)
    postaplot.postaplot(df, x="group", y="y", ax=axs[0], width_distr="normal", seed=1)
    axs[0].set_title("Normal jitter")
    postaplot.postaplot(df, x="group", y="y", ax=axs[1], width_distr="uniform", seed=1)
    axs[1].set_title("Uniform jitter")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "jitter_normal_uniform.png"), dpi=150)
    print("Wrote", os.path.join(OUT, "jitter_normal_uniform.png"))
    plt.close(fig)

def control_jitter_width():
    df = toy_df(100, seed=3)
    fig, axs = plt.subplots(1, 2, figsize=(8,3.2), sharey=True)
    postaplot.postaplot(df, x="group", y="y", ax=axs[0], width_distr="normal", seed=1)
    axs[0].set_title("Default width")
    postaplot.postaplot(df, x="group", y="y", ax=axs[1], width_distr="uniform", seed=1, horizontal_scale=0.05)
    axs[1].set_title("Custom width")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "jitter_width.png"), dpi=150)
    print("Wrote", os.path.join(OUT, "jitter_width.png"))
    plt.close(fig)

def discrete_hue():
    df = toy_df(120, seed=4, dis_hue=True)
    fig, ax = plt.subplots(figsize=(5.6,3.2))
    postaplot.postaplot(df, x="group", y="y", hue="hue", ax=ax, seed=2, reference=True)
    savefig(ax, "discrete_hue.png"); plt.close(fig)

def continuous_hue():
    df = toy_df(150, seed=5, con_hue=True)
    fig, ax = plt.subplots(figsize=(6.2,3.2))
    postaplot.postaplot(df, x="group", y="y", hue="hue_num", ax=ax, reference="auto", seed=1)
    savefig(ax, "continuous_hue.png"); plt.close(fig)

def hue_customization():
    df = toy_df(120, seed=6, dis_hue=True, con_hue=True)
    fig, axs = plt.subplots(1, 2, figsize=(8,3.2), sharey=True)
    postaplot.postaplot(df, x="group", y="y", hue="hue", ax=axs[0], width_distr="normal", seed=1,
    hue_order=['v', 'u'], palette={'u':'r', 'v':'g'})
    axs[0].set_title("Ordered hues, mapping palette")
    postaplot.postaplot(df, x="group", y="y", hue="hue_num", ax=axs[1], width_distr="uniform", seed=1, horizontal_scale=0.05, hue_norm=colors.CenteredNorm(vcenter=0.5), palette='coolwarm')
    axs[1].set_title("Center-normalized hues")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "jitter_custom_hues.png"), dpi=150)
    print("Wrote", os.path.join(OUT, "jitter_custom_hues.png"))
    plt.close(fig)

def hollow_markers():
    df = toy_df(120, seed=6, dis_hue=True)
    fig, ax = plt.subplots(figsize=(5.6,3.2))
    postaplot.postaplot(df, x="group", y="y", hue="hue", dodge=False,
                          ax=ax, hollow=True, markersize=6, alpha=0.8, seed=3)
    savefig(ax, "hollow.png"); plt.close(fig)

def dodge_subgroups():
    df = toy_df(120, seed=8, dis_hue=True)
    fig, ax = plt.subplots(figsize=(5.6,3.2))
    postaplot.postaplot(df, x="group", y="y", hue="hue", dodge=True, ax=ax, alpha=0.5, seed=5)
    savefig(ax, "dodge.png"); plt.close(fig)

def with_box():
    df = toy_df(100, seed=2)
    fig, ax = plt.subplots(figsize=(5,3.2))
    postaplot.postaplot(df, x="group", y="y", ax=ax, seed=0, box=True)
    savefig(ax, "with_box.png"); plt.close(fig)

def with_custom_box():
    df = toy_df(100, seed=2, dis_hue=True)
    fig, ax = plt.subplots()
    postaplot.postaplot(df, x="group", y="y", hue="hue", dodge=True,
                        ax=ax, alpha=0.5, box=True,
                        box_kwa={'color':'r', 'lw':3, 'showfliers':True, 'widths':0.2, 
                        'whiskerprops':{'lw':1.5, 'c':'b'},
                        'boxprops':{'facecolor':'lightgrey', 'alpha':0.7, 'lc':'k'},
                        })
    savefig(ax, "with_custom_box.png"); plt.close(fig)


def supress_legend():
    df = toy_df(100, seed=9, con_hue=True)
    fig, ax = plt.subplots(figsize=(6.2,3.2))
    postaplot.postaplot(df, x="group", y="y", hue="hue_num", ax=ax, reference=False, seed=1)
    savefig(ax, "no_legend.png"); plt.close(fig)

def low_level_engine():
    y = np.random.normal(size=50)
    fig, ax = plt.subplots()
    rng = np.random.default_rng(0)
    coll = postaplot.postaplot_engine(loc=0, series=y, ax=ax, rng=rng)
    print(type(coll))
    savefig(ax, "low_level.png"); plt.close(fig)

def separate_edge_face():
    df = toy_df(120, seed=7, dis_hue=True, con_hue=True)
    plot_seed = 0
    fig, ax = plt.subplots(figsize=(5.6,3.2))
    # plot point faces
    postaplot.postaplot(
        df, x="group", y="y", hue='hue',
        mec='none', palette={'u':"tab:blue", 'v':"tab:green"},
        markersize=7, alpha=0.7, seed=plot_seed
    )
    # plot point edges
    postaplot.postaplot(
        df, x="group", y="y", hue='hue_num',
        mfc='none', palette='managua', mew=2,
        markersize=7, alpha=1, seed=plot_seed
    )
    savefig(ax, "separate_edge_face.png"); plt.close(fig)


def main():
    basic_example()
    jitter_normal_vs_uniform()
    control_jitter_width()
    discrete_hue()
    continuous_hue()
    hue_customization()
    hollow_markers()
    dodge_subgroups()
    with_box()
    with_custom_box()
    supress_legend()
    low_level_engine()
    separate_edge_face()
    

if __name__ == "__main__":
    main()