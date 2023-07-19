"""Boxdim submodule"""

import numpy as np
import numpy.typing as npt

from freud.box import Box
from typing import Self


class BoxDim:
    def __init__(self):
        pass

    @classmethod
    def from_freud(
        cls,
        freud_box: Box,
    ) -> Self:
        pass

    @classmethod
    def cube(
        cls,
        l: float,
    ) -> Self:
        pass

    @classmethod
    def from_array(
        cls,
        sbox: npt.NDArray[np.float32],
        periodic: npt.NDArray[np.bool_],
    ) -> Self:
        pass

    @property
    def l(self):  # noqa: E743
        pass

    @property
    def tilt(self):
        pass

    def periodic(self):
        pass

    def is_2d(self):
        pass

    def volume(self):
        pass
