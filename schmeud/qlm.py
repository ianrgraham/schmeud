"""Quasi-localized mode computer"""

from scipy.sparse.linalg import eigsh
import scipy.linalg
import scipy.sparse as ssp
import gsd.hoomd
import numpy as np
from numba import njit
from jax import grad, jit, lax
from typing import Callable
from freud.locality import AABBQuery
from freud.box import Box

import jax
jax.config.update('jax_platform_name', 'cpu')


class TypeParamDict(dict):

    def __init__(self, grad2_pots, grad3_pots, pot_factory):
        self._pot_factory = pot_factory
        self.grad2_pots = grad2_pots
        self.grad3_pots = grad3_pots
        super().__init__()

    def __setitem__(self, key, value):
        # convert to alpha
        key = tuple(sorted([ord(k.upper()) - 65 for k in key]))
        # key = tuple(sorted(key))
        # M_2 = D^2{ f(x) } = H
        g2 = jit(grad(jit(grad(self._pot_factory(**value)))))
        g3 = jit(grad(g2))  # M_3 = D{ H }
        self.grad2_pots[key] = g2
        self.grad3_pots[key] = g3
        super().__setitem__(key, value)


class Pair:
    """Pair potential from which quasi-localized modes may be calculated.

    This class leverages the auto-grad tool `jax` to automatically differentiate
    and JIT compile a given pair potential, enabling fast construction of the
    Hessian (dynamical matrix) to compute eigenvalues and eigenvectors.

    Arguments
    ---------
    - `pot_factory`: `Callable` - Factory function that takes in a variable number of
    parameters as arguments and returns a `jax` compatible pair potential of the
    type signature `(float) -> float`.

    Example
    -------
    ``` python

    def pot_factory(sigma):

        # define pair potential compatible with `jax`
        def _lambda(x):
            term = 1-x/sigma
            return 0.4/(sigma)*lax.sqrt(term)*(term)

        # handle any conditional using `lax.cond`
        return lambda x: (
            lax.cond(
                x < sigma, 
                _lambda, 
                lambda x: 0.0,
                x
            )
        )

    pair = Pair(pot_factory)

    pair.params[("A","A")] = dict(sigma=14/12)
    pair.params[("A","B")] = dict(sigma=1.0)
    pair.params[("B","B")] = dict(sigma=10/12)

    ```
    """

    def __init__(self, pot_factory: Callable):
        self._pot_factory = pot_factory
        self._cutoffs = {}
        self._grad2_pots = {}
        self._grad3_pots = {}

        self._typeparam_dict = TypeParamDict(
            self._grad2_pots,
            self._grad3_pots,
            self._pot_factory
        )

    @property
    def params(self):
        return self._typeparam_dict

    @property
    def cutoffs(self):
        return self._cutoffs

    def max_cutoff(self):
        max_cut = 0.0
        for cutoff in self._cutoffs.values():
            max_cut = max(max_cut, cutoff)
        return max_cut


class BidispHertz(Pair):

    def __init__(self):

        # it's necessary to have function that generates our pair potential
        # between types given a set of parameters
        def hertzian(sigma):

            # define pair potential compatible with `jax`
            def _lambda(x):
                term = 1-x/sigma
                return 0.4/(sigma)*lax.sqrt(term)*(term)

            # handle any conditional using `lax.cond`
            return lambda x: (
                lax.cond(
                    x < sigma,
                    _lambda,
                    lambda x: 0.0,
                    x
                )
            )

        super().__init__(hertzian)

        # connect pair dictionary definitions and lazily produce hessian and
        # mode-filtering gradient functions
        self.params[("A", "A")] = dict(sigma=14/12)
        self.params[("A", "B")] = dict(sigma=1.0)
        self.params[("B", "B")] = dict(sigma=10/12)

        self.cutoffs[("A", "A")] = 14/12
        self.cutoffs[("A", "B")] = 1.0
        self.cutoffs[("B", "B")] = 10/12


class KobAndersenLJ(Pair):

    def __init__(self):

        def lj(epsilon, sigma):

            # define pair potential compatible with `jax`
            def _lambda(r):
                x = sigma/r
                x2 = x*x
                x4 = x2*x2
                x6 = x4*x2
                return 4*epsilon*(x6*x6 - x6)

            return _lambda

        super().__init__(lj)

        # connect pair dictionary definitions and lazily produce hessian and
        # mode-filtering gradient functions
        self.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
        self.params[("A", "B")] = dict(epsilon=1.5, sigma=0.8)
        self.params[("B", "B")] = dict(epsilon=0.5, sigma=0.88)

        self.cutoffs[("A", "A")] = 2.5
        self.cutoffs[("A", "B")] = 2.5*0.8
        self.cutoffs[("B", "B")] = 2.5*0.88


# NOTE ATM this function accepts a dense matrix as the hessian.
# It would be a whole lot more memory efficient if we used a sparse representation
@njit
def _compute_dense_hessian(edges, grad2_us, edge_vecs, dim, hessian):

    # loop over all edges in the system
    for edge_idx in np.arange(len(edges)):

        # don't forget the prefactor of 1/2 from overcounting
        k_vec = 0.5*grad2_us[edge_idx]*edge_vecs[edge_idx]
        k_outer = np.outer(k_vec, k_vec)

        # loop over all combinations of the particles relating to the current edge
        for i in edges[edge_idx]:
            for j in edges[edge_idx]:
                if i == j:
                    hessian[i*dim:(i+1)*dim, j*dim:(j+1)*dim] += k_outer
                else:
                    hessian[i*dim:(i+1)*dim, j*dim:(j+1)*dim] -= k_outer


# TODO implement this function to replace the dense representation
@njit
def _compute_sparse_hessian(edges, grad2_us, edge_vecs, dim):

    # some ideas: it might be more performant to use a hashmap to store the
    # contributions to the upper triangle and to the diagonal

    # scratch that, no need to use a hashmap

    # diagonal terms is an array of known size

    # upper triange can be computed with the current pair indices, with the same size
    # as "edges"

    # maybe we can then construct the CSR matrix internals here?

    pass


@njit
def _tensor_dot(v: np.ndarray, p: np.ndarray):
    """`v` is a rank-3 tensor of shape (l,n,m) and `p` is a rank-2 tensor of shape (n,m).
    This function contracts along indices n & m, resulting in a vector of size (l). """
    shape = v.shape
    out = np.zeros(shape[0])
    for i in np.arange(shape[1]):
        for j in np.arange(shape[2]):
            out += v[:, i, j]*p[i, j]
    return out


@njit
def _filter_mode(vec, edges, u3s, v3s, dim, N):
    # the inner workings of this function are a little confusing.
    # the math is essentially contracting the input `vec`
    # (first transformed into a rank-2 tensor with an outer poduct)
    # with a rank-3 tensor that is constructed from
    # `u3s` and `v3s`. `u3s` are the 3rd-order radial derivates
    # of the pair potential, while `v3s` are the rank-3 tensor product
    # of the unit vector separating particles `i` and `j` found in an
    # edge

    # allocate space for the post-filtration vector
    filt_vec = np.zeros_like(vec)

    # perform a tensor product along the input vector `vec`
    self_outers = np.zeros((N, dim, dim))
    for idx in range(N):
        u1 = vec[idx*dim:(idx+1)*dim]
        self_outers[idx] = np.outer(u1, u1)

    # now loop over edges, contracting a rank-3 tensor with the above tensor product
    # to get out the filtered vec
    for idx in np.arange(edges.shape[0]):
        edge = edges[idx]
        grad3_u = u3s[idx]
        v = v3s[idx]
        part_i = edge[0]
        part_j = edge[1]

        # everything below is basically unreadable tensor math
        u1 = vec[part_i*dim:(part_i+1)*dim]
        u2 = vec[part_j*dim:(part_j+1)*dim]

        v1 = self_outers[part_i]
        v2 = np.outer(u1, u2)
        v3 = self_outers[part_j]

        t1 = _tensor_dot(v, v1)
        t2 = _tensor_dot(v, v2)
        t3 = _tensor_dot(v, v3)

        out = grad3_u*(t1 - 2*t2 + t3)  # and we finally have the answer!

        filt_vec[part_i*dim:(part_i+1)*dim] += out
        filt_vec[part_j*dim:(part_j+1)*dim] -= out

    return filt_vec


class QLM():
    """Computes the quasi-localized modes for a glassy configuration."""

    def __init__(self, pair: Pair):
        # nothing else to really do here.
        # NOTE we could instead keep a list of interactions (in place of a single pair-wise interaction).
        # Then we could compute the QLMs for system that have combinations of bonded, non-bonded,
        # anisotropic, diheadral, etc.
        self._pair = pair

    def _compute_2gs(self, edges, dists, types) -> np.ndarray:

        grad2_us = np.zeros_like(dists)

        for idx, (edge, dist) in enumerate(zip(edges, dists)):
            type_i = types[edge[0]]
            type_j = types[edge[1]]
            grad2_u = self._pair._grad2_pots[tuple(sorted([type_i, type_j]))]
            grad2_us[idx] = grad2_u(dist)

        return grad2_us

    def _compute_3gs(self, edges, unit_vecs, dists, types, dim) -> np.ndarray:

        grad3_us = np.zeros_like(dists)

        grad3_ts = np.zeros((len(dists), dim, dim, dim))

        for idx, (edge, dist, vec) in enumerate(zip(edges, dists, unit_vecs)):
            type_i = types[edge[0]]
            type_j = types[edge[1]]
            grad3_u = self._pair._grad3_pots[tuple(sorted([type_i, type_j]))]

            grad3_us[idx] = grad3_u(dist)

            grad3_ts[idx] = np.tensordot(vec, np.outer(vec, vec), axes=0)

        return grad3_us, grad3_ts

    def _compute_nlist(self, system):
        max_cutoff = self._pair.max_cutoff()

        query_args = dict(mode='ball', r_max=max_cutoff, exclude_ii=True)

        aq = AABBQuery.from_system(system)
        nlist = aq.query(aq.points, query_args).toNeighborList()

        edges = nlist[:]
        dists = nlist.distances[:]

        return edges, dists

    def _compute_uvecs(self, pos, edges, dists, box, dim):
        unit_vecs = np.zeros((len(edges), dim))
        for idx, (i, j) in enumerate(edges):
            unit_vecs[idx] = box.wrap(pos[j] - pos[i])[:dim]/dists[idx]
        return unit_vecs

    def compute(self, system: gsd.hoomd.Snapshot, k=10, filter=True, sigma=0, dense=False):
        """WARNING: Only use dense=True on small systems. """

        dim = system.configuration.dimensions
        N = system.particles.N
        box = Box.from_box(system.configuration.box)
        pos = system.particles.position
        types = system.particles.typeid

        edges, dists = self._compute_nlist(system)

        # TODO need to run a pass on the computed Hessian submatrices and ensure no
        # particles are rattlers (at least dim+1 contacts)

        # NOTE this can probably be refactored to remove the for loop
        # use numba to compute array of naive dist_vecs for all pairs,
        # then pass this entire array to the freud.Box to wrap
        unit_vecs = self._compute_uvecs(pos, edges, dists, box, dim)

        grad2_us = self._compute_2gs(edges, dists, types)

        # now lets construct the hessian and convert it to a sparse replresentation
        # NOTE I really should look into a more memory efficient approach. For large
        # systems the matrix might take up 10-100s of MB or more.
        hessian_dense = np.zeros((N*dim, N*dim))
        _compute_dense_hessian(edges, grad2_us, unit_vecs, dim, hessian_dense)
        hessian_csr = ssp.csr_matrix(hessian_dense)
        # del hessian_dense, grad2_us

        if dense:
            eig_vals, eig_vecs = scipy.linalg.eigh(hessian_dense)
            eig_vecs = list(eig_vecs.T)
        else:
            eig_vals, eig_vecs = eigsh(hessian_csr, k=k, sigma=sigma)
            eig_vecs = list(eig_vecs.T)

        if filter:

            grad3_us, grad3_ts = self._compute_3gs(
                edges, unit_vecs, dists, types, dim)

            filtered_vecs = [
                _filter_mode(v, edges, grad3_us, grad3_ts, dim, N)
                for v in eig_vecs
            ]

            reshaped_vecs = [v.reshape(N, dim) for v in filtered_vecs]

            del grad3_us, grad3_ts

            return eig_vals, reshaped_vecs

        else:
            return eig_vals, eig_vecs
