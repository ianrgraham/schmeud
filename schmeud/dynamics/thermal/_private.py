import numpy as np

# from numba import njit


# @njit
# def p_hop_calc(
#         r_A: np.ndarray,
#         r_B: np.ndarray
# ) -> np.ndarray:
#     mean_A = np.zeros_like(r_A[0])
#     mean_B = np.zeros_like(r_B[0])
#     for i in range(len())


# @njit
def p_hop_interal(
        pos: np.ndarray,
        tr_frames: int
) -> np.ndarray:
    """Fast implementation of phop using numba.

    Scans through the array of postions and calculates phop with the given t_r.

    Arguments
    ---------
    * pos: 3D ndarray of particle coordinates.
    * tr_frames: Size of the scanning window, t_r. Actual calculation will use
        one additional frame at the beginning of the interval.

    Returns
    -------
    * p_hop: 2D ndarray quantifying dynamical activity.
    """

    n_frames = len(pos)
    half = int(tr_frames/2)

    phop = np.zeros((n_frames - tr_frames, len(pos[0])))

    for i in range(len(phop)):
        r_A = pos[i:i+half+1]
        r_B = pos[i+half:i+tr_frames+1]

        # phop[i] = p_hop_calc(r_A, r_B)

        phop[i] = np.sqrt(
            np.mean(np.sum(np.square(r_A - np.mean(r_B, axis=0)), axis=-1), axis=0) *
            np.mean(np.sum(np.square(r_B - np.mean(r_A, axis=0)), axis=-1), axis=0)
        )

    return phop
