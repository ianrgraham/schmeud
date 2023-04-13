"""Extension module implemented in Rust.

This module exposes a number of fast functions for use in Python.
"""

from . import dynamics
from . import ml
from . import statics

__all__ = ["dynamics", "ml", "statics"]