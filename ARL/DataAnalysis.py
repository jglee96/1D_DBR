import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pixelDBR

TRAIN_PATH = 'D:/1D_DBR/trainset/04'
Nfile = 1

tarwave = 300
minwave = 150
maxwave = 3000
wavestep = 5
wavelength = np.array([np.arange(minwave, maxwave, wavestep)])
bandwidth = 50


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

    Nsample = sX.shape[0]

    return sX, sY, Nsample


def Tstatic(pav, Nsample):
    tstack = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(Nsample):
        if pav[i] >= 0 and pav[i] < 0.1:
            tstack[0] += 1
        elif pav[i] >= 0.1 and pav[i] < 0.2:
            tstack[1] += 1
        elif pav[i] >= 0.2 and pav[i] < 0.3:
            tstack[2] += 1
        elif pav[i] >= 0.3 and pav[i] < 0.4:
            tstack[3] += 1
        elif pav[i] >= 0.4 and pav[i] < 0.5:
            tstack[4] += 1
        elif pav[i] >= 0.5 and pav[i] < 0.6:
            tstack[5] += 1
        elif pav[i] >= 0.6 and pav[i] < 0.7:
            tstack[6] += 1
        elif pav[i] >= 0.7 and pav[i] < 0.8:
            tstack[7] += 1
        elif pav[i] >= 0.8 and pav[i] < 0.9:
            tstack[8] += 1
        elif pav[i] >= 0.9 and pav[i] < 1.0:
            tstack[9] += 1

    return tstack


def main():
    X, Y, Nsample = getData()
    reward = np.reshape(pixelDBR.reward(Y, tarwave, wavelength, bandwidth), newshape=(-1, 1))

    Pt = Tstatic(reward, Nsample)

    x = np.arange(10)
    xs = np.arange(0.1, 1.1, 0.1)
    Tvalues = ['0,1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    plt.bar(x, Pt)
    plt.xticks(x, Tvalues)
    plt.show()

    with open('D:/1D_DBR/Figure_plot/Data_distribution(04).csv', "a") as sf:
        np.savetxt(sf, np.reshape(xs, (1, xs.shape[0])), fmt='%.1f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/Data_distribution(04).csv', "a") as sf:
        np.savetxt(sf, np.reshape(Pt, (1, len(Pt))), fmt='%d', delimiter=',')


if __name__=="__main__":
    main()