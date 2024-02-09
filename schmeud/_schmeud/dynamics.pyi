"""Functions pertaining to glassy dynamics, computed from particle
trajectories."""

import numpy as np
import numpy.typing as npt

from typing import Tuple


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


def d2min_frame(
    snap1: npt.NDArray[np.float32],
    snap2: npt.NDArray[np.float32],
    query_point_indices: npt.NDArray[np.uintp],
    point_indices: npt.NDArray[np.uintp],
    boxes: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
) -> npt.NDArray[np.float32]:
    pass


def d2min_and_strain_frame(
    snap1: npt.NDArray[np.float32],
    snap2: npt.NDArray[np.float32],
    query_point_indices: npt.NDArray[np.uintp],
    point_indices: npt.NDArray[np.uintp],
    boxes: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    pass


def d2min_const_neigh(
    init_sys,
    final_sys,
    n_neigh: int = 20,
) -> np.ndarray:
    pass
