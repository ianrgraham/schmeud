from schmeud import _schmeud

import freud
from freud.box import Box
import gsd.hoomd
import numpy as np
import timeit

# print(dir(freud_box))

# a = int(10)

traj = gsd.hoomd.open("data/traj.gsd")
freud_box = Box.from_box(traj[0].configuration.box)
# freud_box = Box.cube(15)

box = _schmeud.boxdim.BoxDim.from_freud(freud_box)
print(box.l)

points = traj[0].particles.position

voro = freud.locality.Voronoi()
print("start", len(points))
t = timeit.timeit(lambda: voro.compute((freud_box, points)), number=1)
print(t)
print("end")
out1 = voro.compute((freud_box, points)).nlist

nlist = voro.nlist

print(box.periodic())

for i, j in zip(nlist.query_point_indices, nlist.point_indices):
    x = np.array(box.wrap(points[i] - points[j]), dtype=np.float32)
    y = freud_box.wrap(points[i] - points[j])
    # print(type(x), type(y))
    # print(np.linalg.norm(x), np.linalg.norm(y))
    # print(points[i] - points[j])
    np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)

print("start", len(points))
t = timeit.timeit(lambda: _schmeud.nlist.Voronoi(box, points), number=1)
print(t)
print("end")
out2 = _schmeud.nlist.Voronoi(box, points).py_neighbor_list()
weights = np.array(out2.weights, dtype=np.float32)
orig_weights = np.array(out1.weights, dtype=np.float32)
np.testing.assert_allclose(weights, orig_weights, rtol=1e-5, atol=1e-5)

