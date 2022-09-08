"""
A collection of algorithms to analyze glassy systems.
"""

import gsd.hoomd
from typing import Union

from . import _schmeud as schmeud_rs  # should deprecate this
from . import _schmeud
from . import dynamics
from . import ml
from . import statics
from . import utils
from . import qlm

__all__ = ["_schmeud", "dynamics", "ml", "statics", "utils", "qlm"]

SystemLike = Union[
    gsd.hoomd.Snapshot,
    gsd.hoomd.HOOMDTrajectory,
    gsd.hoomd._HOOMDTrajectoryView
]