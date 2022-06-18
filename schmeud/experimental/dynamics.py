"""Experimental APIs for calculating dynamical quantities

I'd like to have a nice API for computing dynamical quantities like the MSD, SISF, Phop, and D2min

"""

import numpy as np
from numpy import typing as npt

import gsd.hoomd
import freud
from numba import njit

from freud.msd import MSD

from schmeud import SystemLike

class SISF:

    def __init__(self, snap: gsd.hoomd.Snapshot, mode: str = "uniform"):

        # store the unwrapped initial positions, as well as the the box for unwrapping the trajectory later
        self._box = freud.Box(*snap.configuration.box)
        self._init_pos = self._box.unwrap(
            snap.particles.position,
            snap.particles.image
        )

        # we haven't implemented any other modes yet
        assert mode == "uniform"

        self._sisf = None
        self._accum = 0

    def compute(self, system: SystemLike, k: float) -> npt.ArrayLike:

        box = self._box

        self._sisf = np.zeros(len(system))
        self._accum = 1

        # loop over each frame of the simulation, grab positions, unwrap, and compute average SISF at that instant
        for i, snap in enumerate(system):
            assert isinstance(snap, gsd.hoomd.Snapshot)
            pos = box.unwrap(
                snap.particles.position,
                snap.particles.image
            )
            terms = k*np.linalg.norm(pos - self._init_pos, axis=-1)
            sisf_i = np.mean(np.sin(terms)/terms)
            self._sisf[i] = sisf_i
        

    @property
    def sisf(self):
        if self._sisf is None:
            raise RuntimeError("The SISF has not been computed.")
        else:
            return self._sisf

    def alpha_relax_idx(self) -> int:
        if self._sisf is None:
            raise RuntimeError("The SISF has not been computed.")
        else:
            sisf = self._sisf
            thres = np.exp(-1)
            for i, val in enumerate(sisf):
                if val <= thres:
                    return i
            return i + 1
            


def diffusion_coeff(t: npt.ArrayLike[np.float32], msd: npt.ArrayLike[np.float32], dim: int = 2):
    """Compute diffusion coefficient from timeseries `t` and `msd`."""
    params = np.polyfit(t, msd, 1)
    return params[0]/(2*dim)