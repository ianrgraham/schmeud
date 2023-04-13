"""Locality submodule"""

from typing import Optional
import numpy as np
import numpy.typing as npt


def particle_to_grid_cube(
    point: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    l: np.float32,
    bins: np.uintp,
) -> npt.NDArray:
    pass
