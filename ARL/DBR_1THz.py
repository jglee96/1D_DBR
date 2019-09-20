import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pixelDBR

N_pixel = 80
dx = 5
nh = 2.092  # SiO2 at 1 THz
nl = 1  # AIr
c = 299792458 * (10**6)

tarwave = 300
minwave = 150
maxwave = 3000
wavestep = 5
wavelength = np.array([np.arange(minwave, maxwave, wavestep)])
bandwidth = 50

# Base data
th = tarwave/(4*nh)
tl = tarwave/(4*nl)
print("======== Design Information ========")
print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
print('Th: {}, Tl: {}'.format(th, tl))

Nfile = 20
TRAIN_PATH = 'D:/1D_DBR/trainset/02'

def getData():
    # Load Training Data
    print("========      Load Data     ========")
    Xarray = []
    Yarray = []
    for nf in range(Nfile):
        sname = TRAIN_PATH + '/state_' + str(nf) + '.csv'
        Xtemp = pd.read_csv(sname, header=None)
        Xtemp = Xtemp.values
        Xarray.append(Xtemp)

        Rname = TRAIN_PATH + '/R_' + str(nf) + '.csv'
        Ytemp = pd.read_csv(Rname, header=None)
        Ytemp = Ytemp.values
        Yarray.append(Ytemp)

    sX = np.concatenate(Xarray)
    sY = np.concatenate(Yarray)

    return sX, sY


def getThickness(s, dx, N_pixel):
    thickness = []
    t = 1
    for x in range(N_pixel - 1):
        if s[x] == s[x + 1]:
            t += 1
        else:
            thickness.append(t * dx)
            t = 1
    
    return thickness


def main():
    X, Y = getData()
    rX = X * np.reshape(pixelDBR.reward(Y, tarwave, wavelength, bandwidth), newshape=(-1, 1))
    rX = np.sum(rX, axis=0)

    minX = np.min(rX)
    rX = rX - minX
    avgX = np.mean(rX)

    result_state = (rX >= avgX).astype(int)
    # result_state = np.reshape(result_state, newshape=N_pixel)
    result_R = pixelDBR.calR(result_state, dx, N_pixel, wavelength, nh, nl)
    result_R = np.reshape(result_R, newshape=(1, wavelength.shape[1]))
    result_reward = pixelDBR.reward(result_R, tarwave, wavelength, bandwidth)
    result_fwhm = pixelDBR.calFWHM(result_R, wavelength, tarwave)
    print('result reward: ', result_reward)
    print('resulr fwhm: ', result_fwhm)

    x = np.reshape(wavelength, wavelength.shape[1])
    result_R = np.reshape(result_R, wavelength.shape[1])
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(x, result_R)

    plt.subplot(2, 1, 2)
    x = (c * (1. / wavelength))
    x = np.reshape(x*(10**-12), wavelength.shape[1])
    plt.plot(x, result_R)

    plt.figure(2)
    lx = np.arange(N_pixel)
    plt.bar(lx, result_state, width=1, color='blue')
    plt.show()

    thickness = getThickness(result_state, dx, N_pixel)
    print(thickness)

if __name__ == "__main__":
    main()
