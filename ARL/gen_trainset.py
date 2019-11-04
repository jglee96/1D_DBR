import numpy as np
import tmm
import matplotlib.pyplot as plt
import time
import pandas as pd
import pixelDBR

TRAIN_PATH = 'D:/1D_DBR/trainset/05'
c = 299792458 * (10**(-6))
N_pixel = 200
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

    Rnorm = np.reshape(Rnorm, newshape=(1, wavelength.shape[1]))
    fwhml, fwhmf = pixelDBR.calBand(Rnorm, wavelength, tarwave, minwave, wavestep, 0.5)
    b99l, b99f = pixelDBR.calBand(Rnorm, wavelength, tarwave, minwave, wavestep, 0.99)
    print("========        Result      ========")
    print('THeory fwhm: {} um, {:.3f} THz'.format(fwhml, fwhmf*10**-12))
    print('THeory 99% width: {} um, {:.3f} THz'.format(b99l, b99f*10**-12))

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

    for i in range(3):
        sname = TRAIN_PATH + '/state_' + str(i+77) + '.csv'
        Rname = TRAIN_PATH + '/R_' + str(i+77) + '.csv'

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
                print('{}th {}step {:.3f}s '.format(i+77, n, time.time() - start))

    print('*****Train Set Prepared*****')


def op_main():
    start = time.time()

    for i in range(50):
        op_state = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        sname = TRAIN_PATH + '/state_' + str(i) + '.csv'
        Rname = TRAIN_PATH + '/R_' + str(i) + '.csv'

        for n in range(Nsample):
            ch_idx = np.random.randint(N_pixel)
            op_state = np.reshape(op_state, N_pixel)
            op_state[ch_idx] = abs(op_state[ch_idx] - 1)

            R = calR(op_state, dx, N_pixel, wavelength, nh, nl)
            op_state = np.reshape(op_state, (1, N_pixel))
            R = np.reshape(R, (1, wavelength.shape[1]))

            with open(sname, "a") as sf:
                np.savetxt(sf, op_state, fmt='%d', delimiter=',')
            with open(Rname, "a") as Rf:
                np.savetxt(Rf, R, fmt='%.5f', delimiter=',')

            if (n) % 100 == 0:
                print('{}th {}step {:.3f}s '.format(i, n, time.time() - start))


def combine():
    PATH1 = 'D:/1D_DBR/trainset/02'
    Nfile1 = 20
    PATH2 = 'D:/1D_DBR/trainset/03'
    Nfile2 = 50

    # Load Training Data 1
    print("========      Load Data1     ========")
    Xarray1 = []
    Yarray1 = []
    for nf in range(Nfile1):
        sname = PATH1 + '/state_' + str(nf) + '.csv'
        Xtemp = pd.read_csv(sname, header=None)
        Xtemp = Xtemp.values
        Xarray1.append(Xtemp)

        Rname = PATH1 + '/R_' + str(nf) + '.csv'
        Ytemp = pd.read_csv(Rname, header=None)
        Ytemp = Ytemp.values
        Yarray1.append(Ytemp)

    print("========      Load Data2     ========")
    Xarray2 = []
    Yarray2 = []
    for nf in range(Nfile2):
        sname = PATH2 + '/state_' + str(nf) + '.csv'
        Xtemp = pd.read_csv(sname, header=None)
        Xtemp = Xtemp.values
        Xarray2.append(Xtemp)

        Rname = PATH2 + '/R_' + str(nf) + '.csv'
        Ytemp = pd.read_csv(Rname, header=None)
        Ytemp = Ytemp.values
        Yarray2.append(Ytemp)

    Xarray = Xarray1 + Xarray2
    Yarray = Yarray1 + Yarray2
    sX = np.concatenate(Xarray)
    sY = np.concatenate(Yarray)

    sname = TRAIN_PATH + '/state_' + str(0) + '.csv'
    Rname = TRAIN_PATH + '/R_' + str(0) + '.csv'

    with open(sname, "a") as sf:
        np.savetxt(sf, sX, fmt='%d', delimiter=',')
    with open(Rname, "a") as Rf:
        np.savetxt(Rf, sY, fmt='%.5f', delimiter=',')

if __name__ == "__main__":
    # Theory()
    main()
    # op_main()
    # combine()
