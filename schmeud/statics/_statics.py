import numpy as np
import pandas as pd

from numba import njit

from .. import utils


@njit
def _digitize_lin(x, arr):

    L = arr[1] - arr[0]

    ub = len(arr) + 1
    lb = 0

    j = int((x-arr[0])//L) + 1
    if j < lb:
        j = lb
    elif j > ub:
        j = ub

    return j


@njit
def _digitize_lin_nearest(x, arr):

    L = arr[1] - arr[0]

    ub = len(arr) + 1
    lb = 0

    j = int((x-arr[0])//L) + 1
    if j < lb:
        j = lb
    elif j > ub:
        j = ub
    elif arr[j+1]-x < x-arr[j]:
        j += 1
    return j


@njit
def get_local_rdf_nearest(dists, bin_edges):
    bins = len(bin_edges) - 1
    out = np.zeros((bins))
    dig = np.zeros(len(dists), dtype=np.int64)
    for i in range(len(dig)):
        dig[i] = _digitize_lin_nearest(dists[i], bin_edges)
    for i in range(len(dig)):
        d = dig[i]
        if d > 0 and d <= len(out):
            out[d-1] += 1
    return out


@njit
def get_local_rdf(dists, bin_edges):
    bins = len(bin_edges) - 1
    out = np.zeros((bins))
    dig = np.zeros(len(dists), dtype=np.int64)
    for i in range(len(dig)):
        dig[i] = _digitize_lin(dists[i], bin_edges)
    for i in range(len(dig)):
        d = dig[i]
        if d > 0 and d <= len(out):
            out[d-1] += 1
    return out


def build_rdf_by_softness_for_frame(snap, partitions, r_max=5.0, bins=100, separate_types=False):
    
    rho = 1.2
    bin_edges = np.linspace(0, r_max, bins+1)
    dr = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + dr*0.5
    div = 4*np.pi*bin_centers*bin_centers*dr*rho
    if separate_types:
        _, dists = utils.gsd.get_nlist_dists(snap, r_max*1.05)
    else:
        _, dists = utils.gsd.get_nlist_dists(snap, r_max*1.05)
    outs = {}
    for key, part in partitions.items():
        part_dists = {k: v for k, v in dists.items() if k in part}
        out = np.zeros((bins))
        N = len(part_dists)
        for k, v in part_dists.items():
            out += get_local_rdf(np.array(v), bin_edges)
        out /= div
        outs[key] = [N, out]
    return outs


def build_rdf_by_softness_for_traj(
        traj,
        orig_df: pd.DataFrame,
        cuts,
        r_max=5.0,
        bins=100,
        quiet=False):

    df = orig_df.copy()
    df["cuts"] = pd.cut(df["softness"], cuts)
    frame_group = df.groupby("frames")
    first = True
    for idx in sorted(frame_group.groups.keys()):
        if not quiet:
            if idx%100 == 0:
                print(idx)
        snap = traj[idx]
        tdf = frame_group.get_group(idx)
        gpby = tdf.groupby("cuts", as_index=False)
        partitions = {key.mid: grp.ids.values for (key, grp) in gpby}
        rdfs = build_rdf_by_softness_for_frame(snap, partitions, r_max=r_max, bins=bins)
        if first:
            outs = rdfs
            first = False
        else:
            for key in rdfs.keys():
                outs[key][0] += rdfs[key][0]
                outs[key][1] += rdfs[key][1]
        
    
    return outs