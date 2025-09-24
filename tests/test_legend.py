import matplotlib.pyplot as plt
from postaplot.legend import build_discrete_legend_handles, add_continuous_colorbar

def test_build_discrete_legend_handles_hollow():
    levels = ["u","v","w"]
    pal = {"u":"C0","v":"C1","w":"C2"}
    handles = build_discrete_legend_handles(levels, pal, hollow=True, marker="o", markersize_pt=7)
    assert len(handles) == 3
    # face should be 'none' when hollow
    assert handles[0].get_markerfacecolor() == "none"

def test_add_continuous_colorbar_creates_cbar_axes():
    fig, ax = plt.subplots()
    # simple fake cmap/norm
    from matplotlib import cm, colors
    cmap = cm.get_cmap("viridis")
    norm = colors.Normalize(0, 1)
    cbar = add_continuous_colorbar(ax, cmap, norm, label="hue")
    assert cbar.ax.get_ylabel() == "hue"
    assert len(fig.axes) >= 2
