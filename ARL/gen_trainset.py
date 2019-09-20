import numpy as np
import tmm
import matplotlib.pyplot as plt
import time
import pandas as pd

TRAIN_PATH = 'D:/1D_DBR/trainset/02'
c = 299792458 * (10**(-6))
N_pixel = 80
dx = 5
nh = 2.092
nl = 1
tarwave = 300
minwave = 150
maxwave = 3000
wavestep = 5
wavelength = np.array([np.arange(minwave, maxwave, wavestep)])
Nsample = 1000


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


def Theory():
    th = tarwave/(4*nh)
    tl = tarwave/(4*nl)
    print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
    print('Th: {}, Tl: {}'.format(th, tl))

    n_list = [1, nh, nl, nh, nl, nh, nl, nh, 1]
    d_list = [np.inf, th, tl, th, tl, th, tl, th, np.inf]

    start = time.time()
    Rnorm = []
    for w in wavelength[0]:
        Rnorm.append(tmm.coh_tmm('s', n_list, d_list, 0, w)['R'])
    print("time: ", time.time() - start)

    plt.figure(1)
    x = np.reshape(wavelength, wavelength.shape[1])
    Rnorm = np.reshape(Rnorm, wavelength.shape[1])
    plt.plot(x, Rnorm)

    plt.figure(2)
    x = (c * (1. / wavelength))
    x = np.reshape(x, wavelength.shape[1])
    Rnorm = np.reshape(Rnorm, wavelength.shape[1])
    plt.plot(x, Rnorm)
    plt.show()


def main():
    print('N_pixel: {}, nh: {:.3f}, nl: {:.3f}'.format(N_pixel, nh, nl))
    start = time.time()

    for i in range(10):
        sname = TRAIN_PATH + '/state_' + str(i+10) + '.csv'
        Rname = TRAIN_PATH + '/R_' + str(i+10) + '.csv'

        for n in range(Nsample):
            state = np.random.randint(2, size=N_pixel)

            R = calR(state, dx, N_pixel, wavelength, nh, nl)
            state = np.reshape(state, (1, N_pixel))
            R = np.reshape(R, (1, wavelength.shape[1]))

            with open(sname, "a") as sf:
                np.savetxt(sf, state, fmt='%d', delimiter=',')
            with open(Rname, "a") as Rf:
                np.savetxt(Rf, R, fmt='%.5f', delimiter=',')

            if (n) % 100 == 0:
                print('{}th {}step {:.3f}s '.format(i+10, n, time.time() - start))

    print('*****Train Set Prepared*****')

if __name__ == "__main__":
    # Theory()
    main()
