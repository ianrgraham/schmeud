"""Functions helpful in computing structure functions for ML models."""

import numpy as np
import numpy.typing as npt


def get_rad_sf_frame(
    nlist_i: npt.NDArray[np.uint32],
    nlist_j: npt.NDArray[np.uint32],
    drs: npt.NDArray[np.float32],
    type_ids: npt.NDArray[np.uint8],
    types: np.uint8,
    mus: npt.NDArray[np.float32],
    spread: np.uint8
) -> npt.NDArray:
    pass


def radial_sf_snap_generic_nlist(
    query_point_indices: npt.NDArray[np.uint32],
    point_indices: npt.NDArray[np.uint32],
    neighbor_counts: npt.NDArray[np.uint32],
    segments: npt.NDArray[np.uint32],
    distances: npt.NDArray[np.float32],
    type_id: npt.NDArray[np.uint8],
    types: np.uint8,
    mus: npt.NDArray[np.float32],
    spread: np.uint8
) -> npt.NDArray:
    pass


def get_rad_sf_frame_subset(
    nlist_i: npt.NDArray[np.uint32],
    nlist_j: npt.NDArray[np.uint32],
    drs: npt.NDArray[np.float32],
    type_ids: npt.NDArray[np.uint8],
    types: np.uint8,
    mus: npt.NDArray[np.float32],
    spread: np.uint8,
    subset: npt.NDArray[np.uint32]
) -> npt.NDArray:
    pass
