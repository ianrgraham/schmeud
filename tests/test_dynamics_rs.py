import numpy as np
from numpy.linalg import inv

import pytest

from schmeud._schmeud import dynamics  # type: ignore

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
    """Test d2min implementation in rust and python against known true results
    """

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
    sol = np.array([[ 3.74193548,  0.64516129],
                    [-1.83870968,  0.09677419]])
    ans = pytest.approx(np.sum(np.square(b0.dot(sol) - b)))
    assert ans1 == ans