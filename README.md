# schmeud *(working title)

## Overview

The **schmeud** library, inspired by [freud](https://github.com/glotzerlab/freud), is a collection of tools for analyzing molecular dynamics trajectories, but with a particular focus towards the study of granular materials, polymers, supercooled liquids, and other glassy systems. This library, like **freud**, aims to be performant yet easy to use as a Python library. To achieve this we write core functionality in the [Rust](https://www.rust-lang.org/) programming language and use [pyo3](https://github.com/PyO3/pyo3) to interop between Rust and Python.

## Installation

Installing **schmeud** is easy with **pip**

``` bash 
pip install schmeud
```

## Usage

``` python
from schmeud.ml import SoftnessDescriptor
from schmeud.filter import Type, Tag
from schmeud.dynamics import Phop

import gsd.hoomd
import numpy as np

rads = np.linspace(0.1, 3.0, 30)
mu = 0.1
types = ['A', 'B']

soft_desc = SoftnessDescriptor.parrinello_radial(rads, mu, types)

traj = gsd.hoomd.open("my-sim.gsd")
filter = Type('A')

phop = Phop(tr=11)

phop_result = phop.compute(system=traj, filter=filter)
training_mask = np.logical_and(phop_result > 0.2, phop_result < 0.05)
training_classes = np.where(phop_result[training_mask] > 0.2, 1, 0)
training_filter = Tag(training_mask)

struc_funcs = soft_desc.compute_sf(system=traj, filter=training_filter)

soft_desc.train(struc_funcs, training_classes)

soft_desc.print_metrics()
```