import numpy as np
import tmm

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


def calFWHM(R,wavelength,tarwave):
    taridx = np.where(wavelength == tarwave)[1][0]
    tarint = R[0, taridx]
    
    tarhi = list(i for i in range(taridx,wavelength.shape[1],1) if R[0, i] < 0.5*tarint)[0]
    tarlo = list(i for i in range(taridx,0,-1) if R[0, i] < 0.5*tarint)[0]

    return tarhi - tarlo


def reward(R, tarwave, wavelength, bandwidth):
    lband = tarwave - int(bandwidth / 2)
    uband = tarwave - int(bandwidth / 2)
    lb_idx = np.where(wavelength == lband)[1][0]
    ub_idx = np.where(wavelength == uband)[1][0]

    R_in = np.mean(R[:, lb_idx:ub_idx+1], axis=1)
    R_out = np.mean(np.hstack((R[:, 0:lb_idx+1], R[:, ub_idx:])), axis=1)

    return  R_in * (1 - R_out)
    # return R_in / R_out