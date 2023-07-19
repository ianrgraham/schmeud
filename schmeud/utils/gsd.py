import warnings

from collections import defaultdict
from typing import Tuple, DefaultDict, List

import freud
import freud.locality
import gsd.hoomd

from .._deprecated import deprecated


@deprecated
def get_freud_box(snapshot: gsd.hoomd.Snapshot) -> freud.box.Box:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        box: freud.box.Box = freud.box.Box(
            *snapshot.configuration.box,
            snapshot.configuration.dimensions == 2)
        return box


@deprecated
def get_nlist_dists(
        snapshot: gsd.hoomd.Snapshot,
        r_max: float) -> Tuple[DefaultDict[int, List], DefaultDict[int, List]]:
    nlist_query = freud.locality.AABBQuery.from_system(snapshot)
    nlist = nlist_query.query(snapshot.particles.position, {
        "r_max": r_max,
        "exclude_ii": True
    }).toNeighborList()
    edges: DefaultDict[int, list] = defaultdict(list)
    distances: DefaultDict[int, list] = defaultdict(list)
    for i, j, w in zip(nlist.query_point_indices[:], nlist.point_indices[:],
                       nlist.distances[:]):
        edges[i].append(j)
        distances[i].append(w)
    return edges, distances


@deprecated
def get_nlist(snapshot: gsd.hoomd.Snapshot,
              r_max: float) -> freud.locality.NeighborList:
    nlist_query = freud.locality.AABBQuery.from_system(snapshot)
    nlist: freud.locality.NeighborList = nlist_query.query(
        snapshot.particles.position, {
            "r_max": r_max,
            "exclude_ii": True
        }).toNeighborList()
    return nlist
