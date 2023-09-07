import time
import numpy as np
import gsd.hoomd
import freud
from numpy.linalg import inv

import pytest

from schmeud._schmeud import dynamics  # type: ignore
from schmeud._schmeud import boxdim
import schmeud.core_prelude as cs

TRAJ = "tests/data/traj.gsd"


def d2min_py(b0, b):
    """Calculates D2min for a set of bonds
    Args
        b0: initial bond lengths
        b: final bond lengths
    """
    V = b0.transpose().dot(b0)
    W = b0.transpose().dot(b)
    J = inv(V).dot(W)
    non_affine = b0.dot(J) - b
    d2min = np.sum(np.square(non_affine))
    return d2min

def test_d2min_class():

    traj = gsd.hoomd.open(name=TRAJ)
    box = traj[0].configuration.box
    box = freud.box.Box.from_box(box)
    pos = traj[0].particles.position

    boxdim.BoxDim.from_freud(box)
    

    b = np.array([[1, 2, 3], [4, 5,6]], dtype=np.float32)

    dynamics.d2min_v2((pos, box), (pos, box), 1)