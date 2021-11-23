from typing import Optional
import numpy as np
import gsd
import gsd.hoomd
import freud

from freud.box import Box
from numpy.linalg import inv
from numba import njit

from ...utils import gsd as util_gsd

def make_box(s: gsd.hoomd.Snapshot):
    dim = s.configuration.dimensions
    box = Box(*s.configuration.box, dim==2)
    return box


@njit
def get_d2min(bonds0: np.ndarray, bonds1: np.ndarray):
    dimension = bonds0.shape[1]
    V = bonds0.transpose().dot(bonds0)
    W = bonds0.transpose().dot(bonds1)
    J = inv(V).dot(W)
    non_affine = bonds0.dot(J) - bonds1
    d2min = np.sum(np.square(non_affine))
    eta = 0.5 * (J * J.transpose() - np.eye(dimension))
    eta_m = 1.0/np.double(dimension) * np.trace(eta)
    tmp = eta - eta_m * np.eye(dimension)
    eta_s = np.sqrt(0.5*np.trace(tmp*tmp))
    return (d2min, J, eta_s)


def get_d2min_config(pos0, pos1, box0, box1, nlist):

    dimension = 2

    d2mins = np.zeros((len(pos0)))
    eta_ss = np.zeros((len(pos0)))
    Js = np.zeros((len(pos0), dimension, dimension))

    for i in np.arange(len(pos1)):
        neighbors = nlist[i]
        bonds0 = np.ascontiguousarray(
            box0.wrap(
                np.array([pos0[j] - pos0[i] for j in neighbors])
            )[:, :dimension])
        bonds1 = np.ascontiguousarray(
            box1.wrap(
                np.array([pos1[j] - pos1[i] for j in neighbors]))[:, :dimension])
        d2min, J, eta_s = get_d2min(bonds0, bonds1)
        d2mins[i] = d2min
        Js[i] = J
        eta_ss[i] = eta_s

    return (d2mins, Js, eta_ss)


def get_D2mins_traj(traj_file: str, slice_: Optional[slice] = None):
    traj = gsd.hoomd.open(name=traj_file, mode="rb")
    D2s = []

    def slice_tuple(slice_):
        return slice_.start, slice_.stop, slice_.step

    if slice_ is None:
        start = 0
        slice_ = slice(1, None)
    else:
        (start, stop, step) = slice_tuple(slice_)
        if start is None:
            start = 0
        if step is None:
            step = 1
        slice_ = slice(start+step, stop, step)

    s0 = traj[start]
    b0 = make_box(s0)
    pos0 = s0.particles.position[:]
    nlist, _ = util_gsd.get_nlist_dists(s0, 2.0)

    for s1 in traj[slice_]:
        b1 = make_box(s1)
        pos1 = s1.particles.position[:]
        d2mins, _, _ = get_d2min_config(pos0, pos1, b0, b1, nlist)
        D2s.append(d2mins)

        s0 = s1
        b0 = b1
        pos0 = pos1
        nlist, _ = util_gsd.get_nlist_dists(s0, 2.0)

    traj.close()
    return D2s


def D2min_plotting_data_generator(traj_file: str, slice_: Optional[slice] = None):
    traj = gsd.hoomd.open(name=traj_file, mode="rb")

    def slice_tuple(slice_):
        return slice_.start, slice_.stop, slice_.step

    if slice_ is None:
        start = 0
        slice_ = slice(1, None)
    else:
        (start, stop, step) = slice_tuple(slice_)
        if start is None:
            start = 0
        if step is None:
            step = 1
        slice_ = slice(start+step, stop, step)

    s0 = traj[start]
    b0 = make_box(s0)
    pos0 = s0.particles.position[:]
    nlist, _ = util_gsd.get_nlist_dists(s0, 2.0)

    types = s0.particles.typeid

    for s1 in traj[slice_]:
        b1 = make_box(s1)
        pos1 = s1.particles.position[:]
        d2mins, _, _ = get_d2min_config(pos0, pos1, b0, b1, nlist)
        d2mins

        yield d2mins, pos1, types

        s0 = s1
        b0 = b1
        pos0 = pos1
        nlist, _ = util_gsd.get_nlist_dists(s0, 2.0)

    traj.close()

# if __name__ == "__main__":

#     import time

#     bond0s = np.random.uniform(-2, 2, size=(10000, 10, 2))

#     bond1s = bond0s + np.random.uniform(-.5, .5, size=(10000, 10, 2))

#     d2mins = np.zeros(10000)

#     start = time.time()

#     for i in range(10000):
#         bonds0 = bond0s[i]
#         bonds1 = bond1s[i]
#         d2min, _, _ = get_d2min(bonds0, bonds1)
#         d2mins[i] = d2min
#     print(time.time() - start )

#     print(d2mins)
