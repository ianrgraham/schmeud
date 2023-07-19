from schmeud import qlm
import gsd.hoomd

import numpy as np

pair = qlm.BidispHertz()
qlm_comp = qlm.QLM(pair)

snap = gsd.hoomd.Snapshot()

snap.particles.position = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                                    [2, 0, 0], [2, 1, 0], [2, 2, 0], [0, 2, 0],
                                    [1, 2, 0]])
snap.particles.N = len(snap.particles.position)
snap.configuration.dimensions = 2
snap.configuration.box = [3, 3, 0, 0, 0, 0]
snap.particles.types = ["A"]
snap.particles.typeid = np.zeros(len(snap.particles.position), dtype=int)

eig_vals, eig_vecs = qlm_comp.compute(snap, k=2)

print(eig_vals)

print(eig_vecs)
