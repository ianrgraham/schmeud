"""Experimental APIs for calculating dynamical quantities

I'd like to have a nice API for computing dynamical quantities like the MSD,
SISF, Phop, and D2min, T1Events

"""

from typing import Optional
import numpy as np
from numpy import typing as npt

import gsd.hoomd
from gsd.hoomd import Snapshot
import freud
# from numba import njit

# re-export MSD from freud, since it works well
# from freud.msd import MSD

from schmeud import SystemLike


class SISF:

    def __init__(self,
                 snap: Optional[gsd.hoomd.Snapshot] = None,
                 mode: str = "uniform"):
        # store the unwrapped initial positions, as well as the the box for
        # unwrapping the trajectory later
        if snap is not None:
            self._box = freud.Box(*snap.configuration.box)
            self._init_pos = self._box.unwrap(snap.particles.position,
                                              snap.particles.image)
            self._ref_set = True
        else:
            self._ref_set = False

        # we haven't implemented any other modes yet
        assert mode == "uniform"

        self._sisf = None
        self._accum = 0

    def compute(self,
                system: SystemLike,
                k: float,
                reset: bool = True) -> npt.ArrayLike:
        # if reset is True, zero sisf
        if reset:
            self._sisf = np.zeros(len(system))
            self._accum = 1
        else:
            self._accum += 1
            assert len(self._sisf) == len(system)

        if self._ref_set:
            ref_box = self._box
            ref_pos = self._init_pos
        else:
            snap = system[0]
            ref_box = freud.Box(*snap.configuration.box)
            ref_pos = self._box.unwrap(snap.particles.position,
                                       snap.particles.image)

        # loop over each frame of the simulation, grab positions, unwrap, and
        # compute average SISF at that instant
        for i, snap in enumerate(system):
            assert isinstance(snap, gsd.hoomd.Snapshot)
            pos = ref_box.unwrap(snap.particles.position, snap.particles.image)
            terms = k * np.linalg.norm(pos - ref_pos, axis=-1)
            sisf_i = np.mean(np.sin(terms) / terms)
            self._sisf[i] = sisf_i

    @property
    def sisf(self):
        if self._sisf is None:
            raise RuntimeError("The SISF has not been computed.")
        else:
            return self._sisf / self._accum

    def alpha_relax_idx(self) -> int:
        if self._sisf is None:
            raise RuntimeError("The SISF has not been computed.")
        else:
            sisf = self._sisf / self._accum
            thres = np.exp(-1)
            for i, val in enumerate(sisf):
                if val <= thres:
                    return i
            return i + 1


class T1Events:

    def __init__(self):
        # What do we need to setup this computation?
        # We could have a few configurable settings, like whether to perform
        # the delaunay triangulation
        # respecting a proxy of particle size, or without
        pass

    def compute(self, snap_i: Snapshot, snap_j: Snapshot):
        pass


class Phop:

    def __init__(self, tr=None):
        assert tr
        self._tr = tr
        pass

    def compute(self):
        pass


class D2min:

    def __init__(self, nquery=None):
        pass

    def compute(self, system: SystemLike):
        pass


def diffusion_coeff(t: npt.ArrayLike[np.float32],
                    msd: npt.ArrayLike[np.float32],
                    dim: int = 2):
    """Compute diffusion coefficient from timeseries `t` and `msd`."""
    params = np.polyfit(t, msd, 1)
    return params[0] / (2 * dim)
