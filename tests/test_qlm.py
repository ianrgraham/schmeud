from schmeud import qlm
import gsd.hoomd

import numpy as np

pair = qlm.BidispHertz()

snap = gsd.hoomd.Snapshot()

N = 18

# TODO build configuration that is properly minimized
snap.position[:] = np.pad(np.random.random((N, 2))*4 - 2, ((0, 0), (0, 1)))
snap.box[:] = [4.0, 4.0, 0.0, 0.0, 0.0, 0.0]
snap.types = ["A", "B"]
snap.typeid[:] = np.zeros(N)
snap.typeid[N/2:] = 1

