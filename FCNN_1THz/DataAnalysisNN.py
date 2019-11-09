import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = 'D:/1D_DBR/trainset/02'
Nfile = 20

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


def score(R, tarwave, wavelength, bandwidth):
    lband = tarwave - int(bandwidth / 2)
    uband = tarwave - int(bandwidth / 2)
    lb_idx = np.where(wavelength == lband)[1][0]
    ub_idx = np.where(wavelength == uband)[1][0]

    R_in = np.mean(R[:, lb_idx:ub_idx+1], axis=1)
    R_out = np.mean(np.hstack((R[:, 0:lb_idx+1], R[:, ub_idx:])), axis=1)

    reward = R_in * (1-R_out)
    cost = R_in * (1-R_in)
    return reward, cost


def reward_static(pav, Nsample):
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

def cost_static(pav, Nsample):
    tstack = [0, 0, 0, 0, 0]
    for i in range(Nsample):
        if pav[i] >= 0 and pav[i] < 0.05:
            tstack[0] += 1
        elif pav[i] >= 0.05 and pav[i] < 0.1:
            tstack[1] += 1
        elif pav[i] >= 0.1 and pav[i] < 0.15:
            tstack[2] += 1
        elif pav[i] >= 0.15 and pav[i] < 0.2:
            tstack[3] += 1
        elif pav[i] >= 0.2 and pav[i] < 0.25:
            tstack[4] += 1

    return tstack


def main():
    X, Y, Nsample = getData()
    reward, cost = score(Y, tarwave, wavelength, bandwidth)
    reward = np.reshape(reward, newshape=(-1, 1))
    cost = np.reshape(cost, newshape=(-1, 1))

    Tcost = cost_static(cost, Nsample)
    Treward = reward_static(reward, Nsample)

    plt.figure(1)
    x = np.arange(10)
    xr = np.arange(0.1, 1.1, 0.1)
    Tvalues = ['0,1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    plt.bar(x, Treward)
    plt.xticks(x, Tvalues)

    plt.figure(2)
    x = np.arange(5)
    xc = np.arange(0.05, 0.3, 0.05)
    Tvalues = ['0.05', '0.10', '0.15', '0.20', '0.25']
    plt.bar(x, Tcost)
    plt.xticks(x, Tvalues)

    with open('D:/1D_DBR/Figure_plot/Data_distribution(02).csv', "a") as sf:
        np.savetxt(sf, np.reshape(xr, (1, xr.shape[0])), fmt='%.2f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/Data_distribution(02).csv', "a") as sf:
        np.savetxt(sf, np.reshape(Treward, (1, len(Treward))), fmt='%d', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/Data_distribution(02).csv', "a") as sf:
        np.savetxt(sf, np.reshape(xc, (1, xc.shape[0])), fmt='%.2f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/Data_distribution(02).csv', "a") as sf:
        np.savetxt(sf, np.reshape(Tcost, (1, len(Tcost))), fmt='%d', delimiter=',')

    plt.show()

if __name__=="__main__":
    main()