"""Quasi-localized mode calculator"""

from abc import ABC, abstractmethod
from typing import Callable
from freud.locality import AABBQuery

import jax.numpy as jxp
from jax import grad


class Pair(ABC):

    def __init__(self, types, jax_potential):
        # set valid types, enumerate to initialize type dictionary
        pass

    def _max_cutoff(self):
        pass

    @property
    def pair(self):
        return self._pair

    @pair.setter
    def pair(self, value):
        if not isinstance(value, Callable):
            raise ValueError
        else:
            self._pair = value

    def v(self, dr, type_i, type_j):
        return self.pair()
    
class Hertzian(Pair):
    pass

class QLM():

    def __init__(self, pair: Pair):
        pass
    
    def compute(self, system):

        query_args = dict(mode='ball', r_min=14/12+0.0001, exclude_ii=True)

        aq = AABBQuery.from_system(system)
        nlist = aq.query(aq.points, query_args)