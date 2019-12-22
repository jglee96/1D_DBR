import numpy as np
import tmm

c = 299792458 * (10**6)


def calR(s, dx, N_pixel, wavelength, nh, nl):
    # list of layer thickness in nm
    d_list = [np.inf] + [dx for i in range(N_pixel)] + [np.inf]
    # list of refractive 
    nht = nh * (s == 1).astype(int)
    nlt = nl * (s == 0).astype(int)
    nt = nht + nlt
    n_list = [1] + nt.tolist() + [1]
    # initialize lists of y-values
    Rnorm = []
    for w in wavelength[0]:
        Rnorm.append(tmm.coh_tmm('s', n_list, d_list, 0, w)['R'])

    return Rnorm


def calBand(R, wavelength, tarwave, minwave, step, ratio):
    taridx = np.where(abs(wavelength - tarwave) < 1e-3)[1][0]
    tarint = R[0, taridx]

    tarhi = minwave + step * list(i for i in range(taridx, wavelength.shape[1], 1) if R[0, i] < ratio * tarint)[0]
    tarlo = minwave + step * list(i for i in range(taridx, 0, -1) if R[0, i] < ratio * tarint)[0]

    length = tarhi - tarlo
    frequency = (c * (1. / tarlo)) - (c * (1. / tarhi))

    return length, frequency


def reward(R, tarwave, wavelength, bandwidth):
    # taridx = np.where(wavelength == tarwave)[1][0]
    # tar_reward = R[0, taridx] * 1e2 - int(R[0, taridx] * 1e2)

    lband = tarwave - int(bandwidth / 2)
    uband = tarwave + int(bandwidth / 2)
    lb_idx = np.where(wavelength == lband)[1][0]
    ub_idx = np.where(wavelength == uband)[1][0]

    R_in = np.mean(R[:, lb_idx:ub_idx+1], axis=1)
    R_out = np.mean(np.hstack((R[:, 0:lb_idx+1], R[:, ub_idx:])), axis=1)

    return R_in * (1 - R_out)