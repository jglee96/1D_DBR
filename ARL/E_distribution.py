import numpy as np
import tmm
import matplotlib.pyplot as plt
import time
import pixelDBR

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


def Theory():
    th = tarwave/(4*nh)
    tl = tarwave/(4*nl)
    print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
    print('Th: {}, Tl: {}'.format(th, tl))

    n_list = [1, nh, nl, nh, nl, nh, nl, nh, nl, nh, 1]
    d_list = [np.inf, th, tl, th, tl, th, tl, th, tl, th, np.inf]
    distance = [i for i in range(int(sum(d_list[1:-1])))]
    low_edge = 230
    high_edge = 430

    start = time.time()
    Rnorm = []
    Ey_low = []
    Ey_high = []
    for w in wavelength[0]:
        tmm_result = tmm.coh_tmm('s', n_list, d_list, 0, w)
        Rnorm.append(tmm_result['R'])
        if w == low_edge:
            for l in distance:
                layer, z = tmm.find_in_structure_with_inf(d_list, l)
                position_result = tmm.position_resolved(layer, z, tmm_result)
                Ey_low.append(position_result['Ey'])
        if w == high_edge:
            for l in distance:
                layer, z = tmm.find_in_structure_with_inf(d_list, l)
                position_result = tmm.position_resolved(layer, z, tmm_result)
                Ey_high.append(position_result['Ey'])
    print("time: ", time.time() - start)

    Rnorm = np.reshape(Rnorm, newshape=(1, wavelength.shape[1]))
    fwhml, fwhmf = pixelDBR.calBand(Rnorm, wavelength, tarwave, minwave, wavestep, 0.5)
    b99l, b99f = pixelDBR.calBand(Rnorm, wavelength, tarwave, minwave, wavestep, 0.99)
    print("========        Result      ========")
    print('Theory fwhm: {} um, {:.3f} THz'.format(fwhml, fwhmf*10**-12))
    print('Theory 99% width: {} um, {:.3f} THz'.format(b99l, b99f*10**-12))

    plt.figure(1)
    x = np.reshape(wavelength, wavelength.shape[1])
    Rnorm = np.reshape(Rnorm, wavelength.shape[1])
    plt.plot(x, Rnorm)
    with open('D:/1D_DBR/Figure_plot/Theory_wave.csv', "a") as sf:
        np.savetxt(sf, np.reshape(x, (1, x.shape[0])), fmt='%d', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/Theory_wave.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Rnorm, (1, Rnorm.shape[0])), fmt='%.5f', delimiter=',')

    plt.figure(2)
    x = (c * (1. / wavelength))
    x = np.reshape(x, wavelength.shape[1])
    Rnorm = np.reshape(Rnorm, wavelength.shape[1])
    plt.plot(x, Rnorm)
    with open('D:/1D_DBR/Figure_plot/THeory_freq.csv', "a") as sf:
        np.savetxt(sf, np.reshape(x, (1, x.shape[0])), fmt='%.5f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/THeory_freq.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Rnorm, (1, Rnorm.shape[0])), fmt='%.5f', delimiter=',')

    plt.figure(3)
    Ey = [abs(y)**2 for y in Ey_low]
    plt.plot(distance, Ey)
    distance = np.array(distance)
    Ey = np.array(Ey)
    with open('D:/1D_DBR/Figure_plot/Theory_E_low.csv', "a") as sf:
        np.savetxt(sf, np.reshape(distance, (1, distance.shape[0])), fmt='%.5f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/Theory_E_low.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Ey, (1, Ey.shape[0])), fmt='%.5f', delimiter=',')

    plt.figure(4)
    Ey = [abs(y)**2 for y in Ey_high]
    plt.plot(distance, Ey)
    distance = np.array(distance)
    Ey = np.array(Ey)
    with open('D:/1D_DBR/Figure_plot/Theory_E_high.csv', "a") as sf:
        np.savetxt(sf, np.reshape(distance, (1, distance.shape[0])), fmt='%.5f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/Theory_E_high.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Ey, (1, Ey.shape[0])), fmt='%.5f', delimiter=',')
    plt.show()


def ARL():
    print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
    # ARL result
    # N_pixel = 80
    pre_n_list = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                  0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    pre_d_list = [dx] * len(pre_n_list)

    # set n_list
    n_list = [1]
    for i in pre_n_list:
        if i == 1:
            n_list.append(nh)
        else:
            n_list.append(nl)
    n_list.append(1)

    # set d_list
    d_list = [np.inf] + pre_d_list + [np.inf]

    distance = [i for i in range(int(sum(d_list[1:-1])))]
    low_edge = 215
    high_edge = 375

    start = time.time()
    Rnorm = []
    Ey_low = []
    Ey_high = []
    for w in wavelength[0]:
        tmm_result = tmm.coh_tmm('s', n_list, d_list, 0, w)
        Rnorm.append(tmm_result['R'])
        if w == low_edge:
            for l in distance:
                layer, z = tmm.find_in_structure_with_inf(d_list, l)
                position_result = tmm.position_resolved(layer, z, tmm_result)
                Ey_low.append(position_result['Ey'])
        if w == high_edge:
            for l in distance:
                layer, z = tmm.find_in_structure_with_inf(d_list, l)
                position_result = tmm.position_resolved(layer, z, tmm_result)
                Ey_high.append(position_result['Ey'])
    print("time: ", time.time() - start)

    Rnorm = np.reshape(Rnorm, newshape=(1, wavelength.shape[1]))
    fwhml, fwhmf = pixelDBR.calBand(Rnorm, wavelength, tarwave, minwave, wavestep, 0.5)
    b99l, b99f = pixelDBR.calBand(Rnorm, wavelength, tarwave, minwave, wavestep, 0.99)
    print("========        Result      ========")
    print('Theory fwhm: {} um, {:.3f} THz'.format(fwhml, fwhmf*10**-12))
    print('Theory 99% width: {} um, {:.3f} THz'.format(b99l, b99f*10**-12))

    plt.figure(1)
    x = np.reshape(wavelength, wavelength.shape[1])
    Rnorm = np.reshape(Rnorm, wavelength.shape[1])
    plt.plot(x, Rnorm)
    with open('D:/1D_DBR/Figure_plot/ARL_wave.csv', "a") as sf:
        np.savetxt(sf, np.reshape(x, (1, x.shape[0])), fmt='%d', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/ARL_wave.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Rnorm, (1, Rnorm.shape[0])), fmt='%.5f', delimiter=',')

    plt.figure(2)
    x = (c * (1. / wavelength))
    x = np.reshape(x, wavelength.shape[1])
    Rnorm = np.reshape(Rnorm, wavelength.shape[1])
    plt.plot(x, Rnorm)
    with open('D:/1D_DBR/Figure_plot/ARL_freq.csv', "a") as sf:
        np.savetxt(sf, np.reshape(x, (1, x.shape[0])), fmt='%.5f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/ARL_freq.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Rnorm, (1, Rnorm.shape[0])), fmt='%.5f', delimiter=',')

    plt.figure(3)
    Ey = [abs(y)**2 for y in Ey_low]
    plt.plot(distance, Ey)
    distance = np.array(distance)
    Ey = np.array(Ey)
    with open('D:/1D_DBR/Figure_plot/ARL_E_low.csv', "a") as sf:
        np.savetxt(sf, np.reshape(distance, (1, distance.shape[0])), fmt='%.5f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/ARL_E_low.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Ey, (1, Ey.shape[0])), fmt='%.5f', delimiter=',')

    plt.figure(4)
    Ey = [abs(y)**2 for y in Ey_high]
    plt.plot(distance, Ey)
    plt.show()
    distance = np.array(distance)
    Ey = np.array(Ey)
    with open('D:/1D_DBR/Figure_plot/ARL_E_high.csv', "a") as sf:
        np.savetxt(sf, np.reshape(distance, (1, distance.shape[0])), fmt='%.5f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/ARL_E_high.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Ey, (1, Ey.shape[0])), fmt='%.5f', delimiter=',')


def ANN():
    print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
    # ARL result
    # N_pixel = 80
    pre_n_list = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                  1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    pre_d_list = [dx] * len(pre_n_list)

    # set n_list
    n_list = [1]
    for i in pre_n_list:
        if i == 1:
            n_list.append(nh)
        else:
            n_list.append(nl)
    n_list.append(1)

    # set d_list
    d_list = [np.inf] + pre_d_list + [np.inf]

    distance = [i for i in range(int(sum(d_list[1:-1])))]
    low_edge = 215
    high_edge = 380

    start = time.time()
    Rnorm = []
    Ey_low = []
    Ey_high = []
    for w in wavelength[0]:
        tmm_result = tmm.coh_tmm('s', n_list, d_list, 0, w)
        Rnorm.append(tmm_result['R'])
        if w == low_edge:
            for l in distance:
                layer, z = tmm.find_in_structure_with_inf(d_list, l)
                position_result = tmm.position_resolved(layer, z, tmm_result)
                Ey_low.append(position_result['Ey'])
        if w == high_edge:
            for l in distance:
                layer, z = tmm.find_in_structure_with_inf(d_list, l)
                position_result = tmm.position_resolved(layer, z, tmm_result)
                Ey_high.append(position_result['Ey'])
    print("time: ", time.time() - start)

    Rnorm = np.reshape(Rnorm, newshape=(1, wavelength.shape[1]))
    fwhml, fwhmf = pixelDBR.calBand(Rnorm, wavelength, tarwave, minwave, wavestep, 0.5)
    b99l, b99f = pixelDBR.calBand(Rnorm, wavelength, tarwave, minwave, wavestep, 0.99)
    print("========        Result      ========")
    print('Theory fwhm: {} um, {:.3f} THz'.format(fwhml, fwhmf*10**-12))
    print('Theory 99% width: {} um, {:.3f} THz'.format(b99l, b99f*10**-12))

    plt.figure(1)
    x = np.reshape(wavelength, wavelength.shape[1])
    Rnorm = np.reshape(Rnorm, wavelength.shape[1])
    plt.plot(x, Rnorm)
    with open('D:/1D_DBR/Figure_plot/ANN_wave.csv', "a") as sf:
        np.savetxt(sf, np.reshape(x, (1, x.shape[0])), fmt='%d', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/ANN_wave.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Rnorm, (1, Rnorm.shape[0])), fmt='%.5f', delimiter=',')

    plt.figure(2)
    x = (c * (1. / wavelength))
    x = np.reshape(x, wavelength.shape[1])
    Rnorm = np.reshape(Rnorm, wavelength.shape[1])
    plt.plot(x, Rnorm)
    with open('D:/1D_DBR/Figure_plot/ANN_freq.csv', "a") as sf:
        np.savetxt(sf, np.reshape(x, (1, x.shape[0])), fmt='%.5f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/ANN_freq.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Rnorm, (1, Rnorm.shape[0])), fmt='%.5f', delimiter=',')

    plt.figure(3)
    Ey = [abs(y)**2 for y in Ey_low]
    plt.plot(distance, Ey)
    distance = np.array(distance)
    Ey = np.array(Ey)
    with open('D:/1D_DBR/Figure_plot/ANN_E_low.csv', "a") as sf:
        np.savetxt(sf, np.reshape(distance, (1, distance.shape[0])), fmt='%.5f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/ANN_E_low.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Ey, (1, Ey.shape[0])), fmt='%.5f', delimiter=',')

    plt.figure(4)
    Ey = [abs(y)**2 for y in Ey_high]
    plt.plot(distance, Ey)
    distance = np.array(distance)
    Ey = np.array(Ey)
    with open('D:/1D_DBR/Figure_plot/ANN_E_high.csv', "a") as sf:
        np.savetxt(sf, np.reshape(distance, (1, distance.shape[0])), fmt='%.5f', delimiter=',')
    with open('D:/1D_DBR/Figure_plot/ANN_E_high.csv', "a") as sf:
        np.savetxt(sf, np.reshape(Ey, (1, Ey.shape[0])), fmt='%.5f', delimiter=',')
    plt.show()

if __name__ == "__main__":
    Theory()
    # ARL()
    # ANN()