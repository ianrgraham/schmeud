"""Internal module wrapping components of the `freud` library.

Intended to be called only from Rust, PLEASE DO NOT USE THIS MODULE!
"""

import freud
import gsd.hoomd


def open(filename: str):
    """Open a hoomd trajectory file."""
    return gsd.hoomd.open(filename)


def open_snap(filename: str, frame: int = 0):
    """Open a hoomd trajectory and extract the snapshot at a given frame."""
    traj = gsd.hoomd.open(filename)
    snap = traj[frame]
    return snap


def box(snap: gsd.hoomd.Snapshot):
    """Get the freud box object to a hoomd snapshot."""
    return freud.box.Box.from_box(snap.configuration.box)


def nlist_query(filename: str, frame: int = 0):
    """Get the nlist query from a hoomd trajectory file."""
    traj = gsd.hoomd.open(filename)
    snap = traj[frame]
    return freud.locality.AABBQuery.from_system(snap)


def nlist_query_snap(snap: gsd.hoomd.Snapshot):
    """Get the nlist query from a hoomd snapshot."""
    return freud.locality.AABBQuery.from_system(snap)
