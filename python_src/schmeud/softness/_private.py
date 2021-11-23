import numpy as np

from numba import njit


@njit
def digitize_lin(x, arr, L):

    ub = len(arr) - 1
    lb = 0

    j = int((x-arr[0])//L)
    if j < lb:
        j = lb
    elif j > ub:
        j = ub
    elif arr[j+1]-x < x-arr[j]:
        j += 1
    return j


@njit
def rad_sf(dist, mu, L):
    '''
        Lightweight implementation of internal softness radial functions
    '''
    term = (dist-mu)/L
    result = np.exp(-term*term*0.5)
    return result


# @njit
# def ang_sf(diff, mu, L):
#     '''
#         Lightweight implementation of internal softness angular functions
#     '''
#     term = (diff-mu)/L
#     result = np.exp(-term*term)
#     return result


@njit
def get_sfs_rad_only(
        diffs,
        labels,
        mus,
        spread):
    '''
        Obtain radial-only softness structure function for a particle in an
        at-most bidisperse ensemble
    '''

    L = mus[1] - mus[0]
    rad_comb = 2
    l_rad = len(mus)

    feature = np.zeros(l_rad*rad_comb)

    idx = 0

    spread_run = spread*2+1

    for i in range(len(diffs)):
        diff = diffs[i]
        label = labels[i]
        j = digitize_lin(diff, mus, L)
        offset = -spread
        nidx = idx + label
        k = j + offset
        for _ in range(spread_run):
            if k < 0 or k >= l_rad:
                k += 1
                continue
            id = nidx + 2*k
            feature[id] += rad_sf(diff, mus[k], L)
            k += 1

    return feature


@njit
def get_sfs_slow_rad_only(
        diffs,
        labels,
        mus):
    rad_comb = 2
    l_rad = len(mus)
    L = mus[1] - mus[0]

    feature = np.zeros(l_rad*rad_comb)

    for i in range(len(diffs)):
        diff = diffs[i]
        label = labels[i]
        sub_feat = rad_sf(diff, mus, L)
        feature[label::2] += sub_feat

    return feature


@njit
def loop_over_features(center_labels, diffs, labels, sf_config):
    '''
        Internal function to loop over all particles
        and fetch softness features
    '''
    mus = np.linspace(
        sf_config.r_min,
        sf_config.r_max,
        int((sf_config.r_max - sf_config.r_min)//(sf_config.r_stride)+1)
    )
    feat_size = len(mus)*2
    spread = sf_config.r_spread

    features = np.zeros((len(center_labels), feat_size))

    for i in range(len(center_labels)):
        features[i] = get_sfs_rad_only(
            diffs[i],
            labels[i],
            mus,
            spread
        )

    return features


@njit
def slow_rad_sf(pos, mu, L):
    result = 0
    for i in np.arange(len(pos)):
        result += np.exp(-np.square((pos[i]-mu)/L))
    return result


@njit
def slow_ang_sf(pos, xi, lam, zeta):
    result = 0
    for j in np.arange(len(pos)-1):
        for k in np.arange(j+1, len(pos)):
            result += np.exp(
                -(pos[j]**2 +
                  pos[k]**2 +
                  np.linalg.norm(pos[k]-pos[j])**2
                  )/xi**2) *\
                np.power(1+lam*pos[k].dot(pos[j]) /
                         (np.linalg.norm(pos[j]) *
                          np.linalg.norm(pos[k])),
                         zeta)
    return result


@njit
def get_sfs_rad_and_ang(
        center_label,
        diffs,
        labels,
        mus,
        spread):
    '''
        Obtain radial and angular softness structure function for a particle in an
        at-most bidisperse ensemble
    '''

    L = mus[1] - mus[0]
    rad_comb = 4
    l_rad = len(mus)

    feature = np.zeros(l_rad*rad_comb)

    idx = 0
    if center_label == 1:
        idx = 2*l_rad

    for i in range(len(diffs)):
        diff = diffs[i]
        label = labels[i]
        j = digitize_lin(diff, mus)
        offset = -spread
        for _ in range(spread*2+1):
            k = j + offset
            if k < 0 or k >= l_rad:
                continue
            feature[idx + 2*k + label] += rad_sf(diff, mus[k], L)

    return feature


@njit
def calc_sam_features(ids, dists, labels, pos, sf_config):

    mus = np.linspace(
        sf_config.r_min,
        sf_config.r_max,
        int((sf_config.r_max - sf_config.r_min)/(sf_config.r_stride)+1)
    )

    ang_feat_size = 50

    rad_feat_size = len(mus)*2
    spread = sf_config.r_spread

    if sf_config.ang:
        features = np.zeros((len(ids), rad_feat_size + ang_feat_size))

    else:
        features = np.zeros((len(labels), rad_feat_size))
