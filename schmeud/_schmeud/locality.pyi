"""Locality submodule"""

import numpy as np
import numpy.typing as npt


def particle_to_grid_cube(
    point: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    l: np.float32,
    bins: np.uintp,
) -> npt.NDArray:
    pass


def particle_to_grid_cube_cic(
    point: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    l: np.float32,
    bins: np.uintp,
) -> npt.NDArray:
    pass

def particle_to_grid_square_cic(
    point: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    l: np.float32,
    bins: np.uintp,
) -> npt.NDArray:
    pass


def particle_to_grid_cube_with_counts(
    point: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    l: np.float32,
    bins: np.uintp,
) -> npt.NDArray:
    pass


def particle_to_grid_cube_cic_with_weights(
    point: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    l: np.float32,
    bins: np.uintp,
) -> npt.NDArray:
    pass


class BlockTree:

    def __init__(
        self,
        grid: npt.NDArray[np.float32],
        periodic: bool
    ):
        pass

    def mass_and_msd(
        self,
        filt: npt.NDArray[np.float32]
    ):
        pass

    def get_sites(
        self,
        filt: npt.NDArray[np.float32]
    ):
        pass
