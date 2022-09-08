import gsd.hoomd
import numpy as np

from .._schmeud import dynamics
from .. import utils

def d2min_frame(
    init_snapshot: gsd.hoomd.Snapshot,
    final_snapshot: gsd.hoomd.Snapshot,
    r_max: float = 2.0
) -> np.ndarray:

    init_pos = init_snapshot.particles.position
    final_pos = final_snapshot.particles.position

    nlist = utils.gsd.get_nlist(init_snapshot, r_max)

    nlist_i = nlist.query_point_indices[:].astype(np.uint32)
    nlist_j = nlist.point_indices[:].astype(np.uint32)

    d2min: np.ndarray = dynamics.d2min_frame(
        init_pos,
        final_pos,
        nlist_i,
        nlist_j
    )

    return d2min