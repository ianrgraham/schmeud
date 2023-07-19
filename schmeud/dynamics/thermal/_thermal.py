from . import _private  # noqa
from ...utils import gsd as gsd_utils

from typing import Optional

import numpy as np
import gsd.hoomd
import scipy.ndimage


def calc_phop(
    traj: gsd.hoomd.HOOMDTrajectory,
    tr_frames: int = 10,
    time_ave_frames: Optional[int] = None,
) -> np.ndarray:
    assert tr_frames % 2 == 0
    assert tr_frames > 0
    assert len(traj) > tr_frames

    n_frames = len(traj)
    init_pos = traj[0].particles.position
    pos = np.zeros((n_frames, *init_pos.shape))
    pos[0] = init_pos
    unwrap = np.zeros(init_pos.shape, dtype=np.int64)
    box = gsd_utils.get_freud_box(traj[0])
    L = box.Lx
    L2 = L / 2.0

    # wrap box positions, should probably refactor this
    # this assumes that the box does not change during the simulation
    for i in range(1, n_frames):
        pos[i] = traj[i].particles.position
        diff = pos[i] - init_pos
        unwrap += np.trunc(diff / L2).astype(np.int64)
        init_pos = pos[i].copy()
        pos[i] -= unwrap.astype(np.float64) * L

    if time_ave_frames is not None:
        pos = scipy.ndimage.uniform_filter1d(pos, time_ave_frames, axis=0)

    phop = _private.p_hop_interal(pos, tr_frames)  # calc p_hop

    return phop
