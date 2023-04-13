import time
import numpy as np
import gsd.hoomd
import freud
from numpy.linalg import inv

import pytest

from schmeud._schmeud import dynamics  # type: ignore

TRAJ = "/media/ian/Data2/monk/oscillatory-mem/workspace/"\
       "413fa9d68e0a7897f131c766528f8182/traj.gsd"


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


def msd_py(x):
    pass


def sisf_py(x, k):
    pass


def test_d2min():
    """Test d2min implementation in rust and python against known results"""

    b0 = np.array([[2], [3]])
    b = np.array([[6], [6]])

    ans1 = d2min_py(b0, b)
    ans = pytest.approx(np.sum(np.square(30/13*b0 - b)))
    assert ans1 == ans

    b0 = np.array([[2, 1],
                   [3, 4],
                   [4, 5]])

    b = np.array([[6, 1],
                  [6, 0],
                  [4, 5]])

    ans1 = d2min_py(b0, b)
    sol = np.array([[3.74193548,  0.64516129],
                    [-1.83870968,  0.09677419]])
    ans = pytest.approx(np.sum(np.square(b0.dot(sol) - b)))
    assert ans1 == ans


def test_d2min_frame():
    """Test that new rust implementation of d2min_frame is correct"""

    traj = gsd.hoomd.open(name=TRAJ)
    for i in range(0, 40, 5):
        snap = traj[i+0]
        snap_later = traj[i+40]

        box = freud.box.Box.from_box(snap.configuration.box)

        sbox = snap.configuration.box[:]
        sbox_later = snap_later.configuration.box[:]
        box_later = freud.box.Box.from_box(snap_later.configuration.box)
        # sbox[2] = 0.0

        nlist_query = freud.locality.LinkCell.from_system(snap)
        nlist = nlist_query.query(
            snap.particles.position,
            {'num_neighbors': 20, "exclude_ii": True}
        ).toNeighborList()

        now = time.time()
        d2min = dynamics.d2min_frame(
            snap.particles.position[:, :2],
            snap_later.particles.position[:, :2],
            nlist.query_point_indices,
            nlist.point_indices,
            (sbox, sbox_later)
        )
        print(time.time() - now)

        now = time.time()
        d2min_truth = []
        iterator = zip(nlist.segments, nlist.neighbor_counts)
        for i, (head, nn) in enumerate(iterator):
            indices = nlist.point_indices[head:head+nn]
            b0 = box.wrap(snap.particles.position[indices]
                          - snap.particles.position[i])[:, :2]
            b = box_later.wrap(snap_later.particles.position[indices]
                               - snap_later.particles.position[i])[:, :2]
            d2min_truth.append(d2min_py(b0, b))

        d2min_truth = np.array(d2min_truth)
        print(time.time() - now)

        # simple timings here show over 10x speedup between rust and python

        np.testing.assert_array_almost_equal(d2min, d2min_truth, decimal=4)


test_d2min_frame()
