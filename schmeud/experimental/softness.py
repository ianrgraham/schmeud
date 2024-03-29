"""Experimental API for building softness calculations and their evaulators

``` python
from schmeud.softness import StrucFuncDescriptor
from schmeud.dynamics import Phop

rads = np.linspace(0.1, 3.0, 30)
mu = 0.1
types = ['A', 'B']
filter = lamda x: x.type == 'A'

sf_desc = StrucFuncDescriptor.parrinello_radial(rads, mu, types)
# StrucFuncDescriptor is also picklable for consistent use within a project

reader = MDAnalysis.coordinates.DCD.DCDReader('trajectory.dcd')

# outputs a pandas DataFrame
struc_funcs = sf_desc.compute(system=reader, topology=topology, filter=filter)

phop = Phop(tr=11)

phop_result = phop.compute(system=reader, topology=topology, filter=filter)

struc_func.to_arrow()
```

"""


class StrucFuncDescriptor:

    def __init__(self):
        self._type = "parrinello"

    @classmethod
    def parrinello_radial(cls, rads, mu, types):
        pass
