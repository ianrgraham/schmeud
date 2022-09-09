"""
A collection of tools to analyze glassy systems.
"""

import gsd.hoomd
from typing import Union

from . import _schmeud
from . import dynamics
from . import ml
from . import statics
from . import utils
from . import qlm
from . import _deprecated

__all__ = ["_schmeud", "dynamics", "ml",
           "statics", "utils", "qlm", "_deprecated"]

SystemLike = Union[
    gsd.hoomd.Snapshot,
    gsd.hoomd.HOOMDTrajectory,
    gsd.hoomd._HOOMDTrajectoryView
]
