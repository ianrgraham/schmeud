"""
User-facing python API of schmeud
"""
import gsd.hoomd
from typing import Union

from . import _schmeud as schmeud_rs
from . import dynamics
from . import ml
from . import statics
from . import utils
from . import qlm

SystemLike = Union[
    gsd.hoomd.Snapshot,
    gsd.hoomd.HOOMDTrajectory,
    gsd.hoomd._HOOMDTrajectoryView
]