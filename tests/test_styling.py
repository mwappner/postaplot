import numpy as np
import pandas as pd

from postaplot.styling import _resolve_hue_mapping, _alias_scatter_kwargs

def test_resolve_hue_mapping_discrete():
    hv = pd.Series(list("aabbccddeeff"))
    is_cont, levels, pal_map, cmap, norm = _resolve_hue_mapping(hv, None, None, None)
    assert not is_cont
    assert set(levels) == set(hv.unique())
    assert set(pal_map.keys()) == set(levels)
    assert cmap is None and norm is None

def test_resolve_hue_mapping_continuous_auto_norm():
    hv = np.linspace(0, 10, 100)
    is_cont, levels, pal_map, cmap, norm = _resolve_hue_mapping(hv, None, None, None)
    assert is_cont
    assert levels == [] and pal_map == {}
    assert cmap is not None and norm is not None
    assert norm.vmin == 0 and norm.vmax == 10

def test_alias_ms_to_s_and_hollow_behavior():
    kw = dict(ms=5, markeredgewidth=1.2, mec="k", hollow=True)
    out = _alias_scatter_kwargs(kw)
    # s should be 25 (area)
    assert "s" in out and (np.atleast_1d(out["s"])[0] == 25)
    # hollow sets facecolors='none'
    assert out.get("facecolors") == "none"
    # edgecolors preserved
    assert out.get("edgecolors") == "k"
