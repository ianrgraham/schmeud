import numpy as np
import pandas as pd

from numba import njit
from scipy.integrate import trapezoid
from .. import _statics
from ... import utils


def s_2_trap(r: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Canonical excess entropy calculated with trapezoidal integration

    Arguments
    ---------
    * r: Bin centers of radius
    * g: Pair correlation function binned by r
    """
    return -trapezoid(np.nan_to_num((g * np.log(g) - g + 1) * r), r)


@njit
def s_2(r: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Canonical excess entropy calculated by summing evenly spaced bins

    Arguments
    ---------
    * r: Evenly spaced bin centers of radius
    * g: Pair correlation function binned by r
    """
    return -np.nansum((g * np.log(g) - g + 1) * r) * (r[1] - r[0])


def local_s2_for_frame(snap,
                       df_orig,
                       r_max=5.0,
                       bins=100,
                       separate_types=False):
    rho = 1.2
    bin_edges = np.linspace(0, r_max, bins + 1)
    dr = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + dr * 0.5
    div = 4 * np.pi * bin_centers * bin_centers * dr * rho

    out_entropy = np.zeros(len(df_orig.index))

    if separate_types:
        _, dists = utils.gsd.get_nlist_dists(snap, r_max * 1.05)
    else:
        _, dists = utils.gsd.get_nlist_dists(snap, r_max * 1.05)
    for idx, k in enumerate(df_orig.ids):
        v = dists[k]
        rdf = _statics.get_local_rdf(np.array(v), bin_edges)
        rdf /= div
        s2 = s_2_trap(bin_centers, rdf)
        out_entropy[idx] = s2
    return out_entropy


def local_s2_for_traj(traj, orig_df, r_max=5.0, bins=50, quiet=False):
    df = orig_df.copy()
    frame_group = df.groupby("frames")
    new_df = []
    for idx in sorted(frame_group.groups.keys()):
        if not quiet:
            if idx % 100 == 0:
                print(idx)
        snap = traj[idx]
        tdf = frame_group.get_group(idx)
        frame_entropies = local_s2_for_frame(snap, tdf, r_max=r_max, bins=bins)
        tdf["entropy"] = frame_entropies
        new_df.append(tdf)

    return pd.concat(new_df)
