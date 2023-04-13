"""Functions pertaining to glassy dynamics, computed from particle
trajectories."""

import numpy as np
import numpy.typing as npt


def nonaffine_local_strain(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32]
) -> np.float32:
    pass


def affine_local_strain_tensor(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    pass


def p_hop(
    traj: npt.NDArray[np.float32],
    tr_frames: np.uintp
) -> npt.NDArray[np.float32]:
    pass
