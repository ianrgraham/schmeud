from ._schmeud import statics as _statics  # noqa: F401
from ._schmeud import ml as _ml  # noqa: F401
from . import utils

from collections import defaultdict, namedtuple
from typing import DefaultDict, Tuple, List, Optional

import numpy as np
import gsd.hoomd
import pandas as pd

from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.signal import find_peaks

import freud

StructureFunctionConfig = namedtuple(
    'StructureFunctionConfig', ['r_min', 'r_max', 'r_stride', "r_spread"])

LinspaceSfConfig = namedtuple("LinSfConfig",
                              ['mu_min', 'mu_max', 'mu_bins', 'mu_spread'])


def compute_structure_functions_snap(
        snap: gsd.hoomd.Snapshot,
        query_indices: Optional[np.ndarray] = None,
        sf_config: Optional[LinspaceSfConfig] = None,
        query_r_min: float = 0.1) -> np.ndarray:

    if query_indices is None:
        query_points = snap.particles.position
    else:
        num_particles = snap.particles.N
        query_array = np.zeros(num_particles, dtype=bool)
        query_array[query_indices] = True
        query_points = snap.particles.position[query_array]

    if sf_config is None:
        sf_config = LinspaceSfConfig(0.4, 3.0, 27, 3)

    mus = np.linspace(sf_config.mu_min,
                      sf_config.mu_max,
                      sf_config.mu_bins,
                      dtype=np.float32)
    # We are going to assume that delta is tied to the bin spacing
    dmu = mus[1] - mus[0]

    nlist_query = freud.locality.AABBQuery.from_system(snap)
    nlist = (nlist_query.query(
        query_points, {
            'r_min': query_r_min,
            'r_max': sf_config.mu_max + dmu * sf_config.mu_spread
        }).toNeighborList())

    sfs = _ml.radial_sf_snap_generic_nlist(
        nlist.query_point_indices, nlist.point_indices,
        nlist.neighbor_counts, nlist.segments, nlist.distances,
        snap.particles.typeid.astype(np.uint8), len(snap.particles.types), mus,
        sf_config.mu_spread)

    return sfs


def calc_structure_functions_dataframe(
        traj: gsd.hoomd.HOOMDTrajectory,
        sf_config: Optional[StructureFunctionConfig] = None,
        dyn_indices: Optional[List[Tuple[int, List[int], List[int]]]] = None,
        sub_slice: Optional[slice] = None,
        flatten: bool = True) -> pd.DataFrame:

    assert (not (sub_slice is not None and dyn_indices is not None))
    # if dyn_indices is None:
    #     assert(dynamics is None)

    features: List[np.ndarray] = []
    meta_frames = []
    meta_ids = []
    meta_labels = []

    if sf_config is None:
        sf_config = StructureFunctionConfig(0.1, 4.0, 0.1, 4)

    nlist_max_r = sf_config.r_max + sf_config.r_stride * sf_config.r_spread

    # two code pathes, the one where dyn_indices is defined should generally be
    # used to fetch features before training, since it also returns truth
    # values
    if dyn_indices is None:

        n_frames = len(traj)

        if sub_slice is None:
            frame_iter = range(n_frames)
        else:
            frame_iter = range(*sub_slice.indices(len(traj)))

        for i in frame_iter:

            snapshot = traj[int(i)]
            nlist_query = freud.locality.AABBQuery.from_system(snapshot)
            nlist = nlist_query.query(snapshot.particles.position, {
                'r_max': nlist_max_r,
                'exclude_ii': True
            }).toNeighborList()
            nlist_i = nlist.query_point_indices[:].astype(np.uint32)
            nlist_j = nlist.point_indices[:].astype(np.uint32)
            drs = nlist.distances[:].astype(np.float32)

            labels = snapshot.particles.typeid.astype(np.uint8)
            types = np.uint8(2)
            points = int((sf_config.r_max - sf_config.r_min) //
                         (sf_config.r_stride) + 1)
            mus = np.linspace(sf_config.r_min,
                              sf_config.r_max,
                              points,
                              dtype=np.float32)
            spread = np.uint8(sf_config.r_spread)

            X = _ml.get_rad_sf_frame(nlist_i, nlist_j, drs, labels, types, mus,
                                     spread)
            assert (len(X) == len(labels))
            if flatten:
                ids = np.arange(snapshot.particles.N)
                meta_frames.extend(list(np.ones_like(ids) * i))
                meta_ids.extend(list(ids))
                meta_labels.extend(list(labels))
                features.extend(list(X))
            else:
                meta_frames.append(i)
                meta_ids.append(np.arange(snapshot.particles.N))
                meta_labels.append(labels)
                features.append(X)

        return pd.DataFrame({
            "frames": meta_frames,
            "ids": meta_ids,
            "labels": meta_labels,
            "Xs": features
        })

    else:

        truths: List = []

        for i, soft, hard in dyn_indices:

            snapshot = traj[int(i)]

            extrema = np.array(hard + soft, dtype=np.uint32)
            t_truths = []
            t_truths.extend([0 for e in hard])
            t_truths.extend([1 for e in soft])

            nlist_query = freud.locality.AABBQuery.from_system(snapshot)
            nlist = nlist_query.query(snapshot.particles.position, {
                'r_max': 1.0,
                'r_min': 0.05
            }).toNeighborList()
            nlist_i = nlist.query_point_indices[:].astype(np.uint32)
            nlist_j = nlist.point_indices[:].astype(np.uint32)
            drs = nlist.distances[:].astype(np.float32)

            labels = np.array(snapshot.particles.typeid).astype(np.uint8)
            types = np.uint8(2)
            mus = np.linspace(sf_config.r_min,
                              sf_config.r_max,
                              int((sf_config.r_max - sf_config.r_min) //
                                  (sf_config.r_stride) + 1),
                              dtype=np.float32)
            spread = np.uint8(sf_config.r_spread)

            X = _ml.get_rad_sf_frame_subset(nlist_i, nlist_j, drs, labels,
                                            types, mus, spread, extrema)

            labels = np.take(labels, extrema)

            assert (len(X) == len(labels))
            if flatten:
                meta_frames.extend(list(np.ones_like(extrema) * i))
                meta_ids.extend(list(extrema))
                meta_labels.extend(list(labels))
                features.extend(list(X))
                truths.extend(t_truths)
            else:
                meta_frames.append(np.ones_like(extrema) * i)
                meta_ids.append(np.array(extrema))
                meta_labels.append(np.array(labels))
                features.append(X)
                truths.append(np.array(t_truths))

        return pd.DataFrame({
            "frames": meta_frames,
            "ids": meta_ids,
            "labels": meta_labels,
            "Xs": features,
            "ys": truths
        })


def train_hyperplane_pipeline(
        X: np.ndarray,
        y: np.ndarray,
        seed: int = 0,
        test_size: float = 0.25,
        max_iter: int = 10_000) -> Tuple[Pipeline, Tuple[float, np.ndarray]]:
    '''Train linear SVM with StandardScaler preprocessing

    Arguments
    ---------
    * X: Training features.
    * y: Training truth values.
    * seed: Random seed to build interal Generator.
    * test_size: Fraction between 0.0 and 1.0 that will be set aside for
        testing.
    * max iter: number of iterations to perform before calling it quits.

    Returns
    -------
    Trained Pipeline object and training accuracy data
    '''

    assert (len(X) == len(y))

    rng = np.random.default_rng(seed)
    rand_seeds = rng.integers(low=0, high=2**32 - 1, size=2)

    shuff_y, shuff_X = shuffle(y, list(X), random_state=rand_seeds[0])

    X_train, X_test, y_train, y_test = train_test_split(shuff_X,
                                                        shuff_y,
                                                        test_size=test_size)

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svc',
                      svm.LinearSVC(class_weight='balanced',
                                    max_iter=max_iter,
                                    random_state=rand_seeds[1]))])

    pipe.fit(X_train, y_train)

    y_predict = pipe.predict(X_test)
    predacs: float = np.sum(y_test == y_predict) / len(y_test)
    confmats = metrics.confusion_matrix(y_test, y_predict) / len(y_test)
    confmats = np.array(confmats)

    print('Prediction accuracy:', predacs, '\nConfusion Matrix:\n', confmats)

    return pipe, (predacs, confmats)


def group_hard_soft_by_cutoffs(
    dynamics: np.ndarray,
    noise_cutoff: float = 0.05,
    rearrange_cutoff: float = 0.2,
    distance: int = 10,
    hard_distance: Optional[int] = None,
    sub_slice: Optional[slice] = None
) -> List[Tuple[int, List[int], List[int]]]:

    # makes building the dyn_list a little easier
    def double_list():
        return [[], []]

    rng = range(sub_slice.start, sub_slice.stop)

    dyn_dict: DefaultDict[int, Tuple[List[int],
                                     List[int]]] = defaultdict(double_list)

    for i in range(dynamics.shape[1]):
        peaks, _ = find_peaks(dynamics[:, i], distance=distance)
        c1 = dynamics[peaks, i] >= rearrange_cutoff
        if hard_distance is None:
            hard_peaks_init = peaks
        else:
            hard_peaks_init, _ = find_peaks(dynamics[:, i],
                                            distance=hard_distance)
        c2 = dynamics[hard_peaks_init, i] < noise_cutoff
        soft_peaks = peaks[c1]
        hard_peaks = hard_peaks_init[c2]
        for p in soft_peaks:
            if p in rng:
                dyn_dict[p][0].append(i)
        for p in hard_peaks:
            if p in rng:
                dyn_dict[p][1].append(i)

    dyn_list = []
    for i in sorted(dyn_dict.keys()):
        dyn_list.append((i, *dyn_dict[i]))

    return dyn_list


def find_soft_particles_by_cutoff(
        dynamics: np.ndarray,
        rearrange_cutoff: float = 0.2,
        distance: int = 10) -> List[Tuple[int, List[int]]]:

    soft_dict = defaultdict(list)

    for i in range(dynamics.shape[1]):
        peaks, _ = find_peaks(dynamics[:, i],
                              height=rearrange_cutoff,
                              distance=distance)
        for p in peaks:
            soft_dict[p].append(i)

    soft_list = []
    for i in sorted(soft_dict.keys()):
        soft_list.append((i, soft_dict[i]))

    return soft_list


def spatially_smeared_local_rdf(
        snapshot: gsd.hoomd.Snapshot,
        r_min: float = 0.05,
        r_max: float = 5.05,
        bins: int = 50,
        collapse_types: bool = False,
        smear_length: Optional[float] = None,
        smear_gauss: Optional[float] = None) -> np.ndarray:

    bin_edges = np.linspace(r_min, r_max, bins + 1)
    dr = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + dr * 0.5
    div = 4 * np.pi * bin_centers * bin_centers * dr

    hull = 4 * np.pi * r_max * r_max * r_max / 3

    nlist_max_r = r_max + smear_gauss * 4.0

    nlist = utils.gsd.get_nlist(snapshot, nlist_max_r)

    nlist_i = nlist.query_point_indices[:].astype(np.uint32)
    nlist_j = nlist.point_indices[:].astype(np.uint32)
    drs = nlist.distances[:].astype(np.float32)

    labels = np.array(snapshot.particles.typeid).astype(np.uint8)
    types = np.uint8(2)

    rdfs = _statics.spatially_smeared_local_rdfs(nlist_i, nlist_j, drs, labels,
                                                 types, r_min, r_max, bins,
                                                 smear_length, smear_gauss)

    if collapse_types:
        rdfs = np.sum(rdfs, axis=2)

        totals = np.sum(rdfs, axis=1)

        rdf_shape = rdfs.shape

        for i in range(rdf_shape[0]):
            rdfs[i, :] /= div

        for i in range(rdf_shape[1]):
            rdfs[:, i] *= hull / totals

    else:
        totals = np.sum(rdfs, axis=1)

        rdf_shape = rdfs.shape

        for i in range(rdf_shape[0]):
            for j in range(rdf_shape[2]):
                rdfs[i, :, j] /= div

        for i in range(rdf_shape[1]):
            rdfs[:, i] *= hull / totals

    return bin_centers, rdfs
