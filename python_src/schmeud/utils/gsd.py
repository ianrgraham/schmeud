import warnings

from collections import defaultdict, namedtuple
from typing import Tuple, DefaultDict, Dict, List

import freud
import freud.locality
import gsd.hoomd
import numpy as np
import numba.typed
import numba.core.types

# from numba import njit


def get_freud_box(snapshot: gsd.hoomd.Snapshot) -> freud.box.Box:

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        box: freud.box.Box = freud.box.Box(
            *snapshot.configuration.box,
            snapshot.configuration.dimensions == 2)
        return box


NeighborList = namedtuple(
    'NeighborList',
    ['size', 'query_point_indices', 'point_indices', 'distances']
)

def get_nlist_dists(
        snapshot: gsd.hoomd.Snapshot,
        r_max: float
) -> Tuple[DefaultDict[int, List], DefaultDict[int, List]]:

    nlist_query = freud.locality.AABBQuery.from_system(snapshot)
    nlist = nlist_query.query(
        snapshot.particles.position,
        {'r_max': r_max, 'exclude_ii': True}).toNeighborList()
    edges: DefaultDict[int, list] = defaultdict(list)
    distances: DefaultDict[int, list] = defaultdict(list)
    for i, j, w in zip(
            nlist.query_point_indices[:],
            nlist.point_indices[:],
            nlist.distances[:]):
        edges[i].append(j)
        distances[i].append(w)
    return edges, distances


def get_nlist_fast(
        snapshot: gsd.hoomd.Snapshot,
        r_max: float
) -> freud.locality.NeighborList:

    nlist_query = freud.locality.LinkCell.from_system(snapshot)
    nlist: freud.locality.NeighborList = nlist_query.query(
        snapshot.particles.position,
        {'r_max': r_max, 'exclude_ii': True}).toNeighborList()
    return nlist


def convert_listdict_to_arraylist(dictionary: Dict[int, List[int]]) -> numba.typed.List:

    out: numba.typed.List = numba.typed.List()
    for i in sorted(dictionary.keys()):
        out.append(numba.typed.List(dictionary[i]))
    return out


def convert_subset_listdict_to_arraylist(
        dictionary: Dict[int, List[int]],
        subset: List[int]) -> numba.typed.List:

    out: numba.typed.List = numba.typed.List()
    for i in subset:
        out.append(numba.typed.List(dictionary[i]))
    return out


def get_label_arraylist(
        nlist: numba.typed.List,
        labels: np.ndarray
) -> numba.typed.List:

    out: numba.typed.List = numba.typed.List()
    for i in range(len(nlist)):
        out.append(numba.typed.List([labels[j] for j in nlist[i]]))
    return out
