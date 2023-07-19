import numpy as np
from scipy.sparse import bsr_matrix


def build_dense(edges, grad2_us, edge_vecs, dim, hessian):

    # loop over all edges in the system
    for edge_idx in np.arange(len(edges)):

        # don't forget the prefactor of 1/2 from overcounting
        k_vec = 0.5 * grad2_us[edge_idx] * edge_vecs[edge_idx]
        k_outer = np.outer(k_vec, k_vec)

        # loop over all combinations of the particles relating to the current
        # edge
        for i in edges[edge_idx]:
            for j in edges[edge_idx]:
                if i == j:
                    hessian[i * dim:(i + 1) * dim,
                            j * dim:(j + 1) * dim] += k_outer
                else:
                    hessian[i * dim:(i + 1) * dim,
                            j * dim:(j + 1) * dim] -= k_outer


def build_csr(edges, grad2_us, edge_vecs, dim, hessian):

    # loop over all edges in the system
    for edge_idx in np.arange(len(edges)):

        # don't forget the prefactor of 1/2 from overcounting
        k_vec = 0.5 * grad2_us[edge_idx] * edge_vecs[edge_idx]
        k_outer = np.outer(k_vec, k_vec)

        # loop over all combinations of the particles relating to the current
        # edge
        for i in edges[edge_idx]:
            for j in edges[edge_idx]:
                if i == j:
                    hessian[i * dim:(i + 1) * dim,
                            j * dim:(j + 1) * dim] += k_outer
                else:
                    hessian[i * dim:(i + 1) * dim,
                            j * dim:(j + 1) * dim] -= k_outer


def build_bsr(edges, grad2_us, edge_vecs, dim, hessian):

    # data = []

    # loop over all edges in the system
    for edge_idx in np.arange(len(edges)):

        # don't forget the prefactor of 1/2 from overcounting
        k_vec = 0.5 * grad2_us[edge_idx] * edge_vecs[edge_idx]
        k_outer = np.outer(k_vec, k_vec)

        # loop over all combinations of the particles relating to the current
        # edge
        for i in edges[edge_idx]:
            for j in edges[edge_idx]:
                if i == j:
                    hessian[i * dim:(i + 1) * dim,
                            j * dim:(j + 1) * dim] += k_outer
                else:
                    hessian[i * dim:(i + 1) * dim,
                            j * dim:(j + 1) * dim] -= k_outer


indptr = np.array([0, 1, 3, 6])

indices = np.array([0, 2, 2, 0, 1, 2, 0])

data = np.array([1, 5, 3, 4, 5, 6, 10]).repeat(4).reshape(7, 2, 2)

print(bsr_matrix((data, indices, indptr), shape=(6, 6)).toarray())
