"""Functions relating to the static structure of particulate systems."""

from typing import Optional
import numpy as np
import numpy.typing as npt


def spatially_smeared_local_rdfs(
    nlist_i: npt.NDArray[np.uint32],
    nlist_j: npt.NDArray[np.uint32],
    drs: npt.NDArray[np.float32],
    type_ids: npt.NDArray[np.uint8],
    types: np.uint8,
    r_min: np.float32,
    r_max: np.float32,
    bins: np.uintp,
    smear_rad: Optional[np.float32],
    smear_gauss: Optional[np.float32]
) -> npt.NDArray:
    pass
