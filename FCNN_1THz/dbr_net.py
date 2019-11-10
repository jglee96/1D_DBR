import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pixelDBR
import argparse
from dbr_core import *

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


def Ratio_Optimization(output_folder, weight_name_save, n_batch, lr_rate, num_layers, n_hidden):
    OUTPUT_SIZE = wavelength.shape[1]

    idx_1 = np.where(wavelength == int(tarwave - bandwidth/2))[1][0]
    idx_2 = np.where(wavelength == int(tarwave + bandwidth/2))[1][0]
    design_y = np.zeros((1, OUTPUT_SIZE))
    design_y[0, idx_1:idx_2+1] = 1
    design_y.tolist()

    init_list_rand = tf.constant(np.random.randint(2, size=(1, N_pixel)), dtype=tf.float32)
    X = tf.get_variable(name='b', initializer=init_list_rand)
    Xint = binaryRound(X)
    Xint = tf.clip_by_value(Xint, clip_value_min=0, clip_value_max=1)
    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name="output_y")
    weights, biases = load_weights(output_folder, weight_name_save, num_layers)
    Yhat = forwardprop(Xint, weights, biases, num_layers)

    idxwidth = idx_2 - idx_1
    Inval = tf.reduce_mean(tf.matmul(Y, tf.transpose(Yhat))) / idxwidth
    Outval = tf.reduce_mean(tf.matmul((1-Y), tf.transpose(Yhat))) / (OUTPUT_SIZE - idxwidth)
    # cost = Outval / Inval
    # cost = Outval * (10 - Inval)
    cost = Outval * (1-Inval)
    optimizer = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(cost, var_list=[X])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n in range(20000):
            sess.run(optimizer, feed_dict={Y: design_y})
            if (n % 100) == 0:
                temp_R = np.reshape(sess.run(Yhat), newshape=(1, wavelength.shape[1]))
                temp_cost = sess.run(cost, feed_dict={Y: design_y})
                temp_Inval = sess.run(Inval, feed_dict={Y: design_y})
                temp_reward = pixelDBR.reward(temp_R, tarwave, wavelength, bandwidth)
                print("{}th epoch, reward: {:.4f}, cost: {:.4f}, Inval: {:.4f}".format(n, temp_reward[0], temp_cost, temp_Inval))
        op_x = np.reshape(Xint.eval().astype(int), newshape=N_pixel)
        # op_R = np.reshape(sess.run(Yhat), newshape=(1, wavelength.shape[1]))
        op_R = np.reshape(pixelDBR.calR(op_x, dx, N_pixel, wavelength, nh, nl), newshape=(1, wavelength.shape[1]))
        op_reward = pixelDBR.reward(op_R, tarwave, wavelength, bandwidth)
        op_fwhml, op_fwhmf = pixelDBR.calBand(op_R, wavelength, tarwave, minwave, wavestep, 0.5)
        op_99l, op_99f = pixelDBR.calBand(op_R, wavelength, tarwave, minwave, wavestep, 0.99)
    print("========        Result      ========")
    print('result fwhm: {} um, {:.3f} THz'.format(op_fwhml, op_fwhmf*10**-12))
    print('result 99% width: {} um, {:.3f} THz'.format(op_99l, op_99f*10**-12))
    print("Optimized result: {:.4f}".format(op_reward[0]))
    for idx, x in enumerate(op_x):
        if idx == 0:
            print("[{}, ".format(x), end='')
        elif idx == N_pixel-1:
            print("{}]".format(x), end='')
        else:
            print("{}, ".format(x), end='')

    wavelength_x = np.reshape(wavelength, wavelength.shape[1])
    op_R = np.reshape(op_R, wavelength.shape[1])
    plt.figure(1)
    plt.plot(wavelength_x, op_R)

    plt.figure(2)
    wavelength_x = (c * (1. / wavelength)) * 10**(-12)
    wavelength_x = np.reshape(wavelength_x, wavelength.shape[1])
    plt.plot(wavelength_x, op_R)

    plt.figure(3)
    pixel_x = np.arange(N_pixel)
    plt.bar(pixel_x, op_x, width=1, color="black")
    plt.show()


def shuffle_data(X, Y):
    Nsample = Y.shape[0]
    x = np.arange(Y.shape[0])
    np.random.shuffle(x)

    X = X[x, :]
    Y = Y[x, :]

    return X, Y


def main(output_folder, weight_name_save, n_batch, lr_rate, num_layers, n_hidden):
    # Load training data
    sX, sY = getData()
    Nsample = sY.shape[0]
    INPUT_SIZE = sX.shape[1]
    OUTPUT_SIZE = sY.shape[1]

    Nr = 0.8
    Nlearning = int(Nr*Nsample)
    Ntest = Nsample - Nlearning

    testX = sX[Nlearning:, :]
    testY = sY[Nlearning:, :]

    trainX = sX[0:Nlearning, :]
    trainY = sY[0:Nlearning, :]
    trainX_total = trainX
    trainY_total = trainY
    n_copy = 10
    for i in range(n_copy):
        trainX, trainY = shuffle_data(trainX, trainY)
        trainX_total = np.concatenate((trainX_total, trainX), axis=0)
        trainY_total = np.concatenate((trainY_total, trainY), axis=0)

    ## Define NN ##
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="input_x")
    Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="output_y")
    weights = []
    biases = []

    for i in range(0, num_layers):
        if i == 0:
            weights.append(init_weights((INPUT_SIZE, n_hidden)))
        else:
            weights.append(init_weights((n_hidden, n_hidden)))
        biases.append(init_bias(n_hidden))
    weights.append(init_weights((n_hidden, OUTPUT_SIZE)))
    biases.append(init_bias(OUTPUT_SIZE))

    # Forward propagation
    Yhat = forwardprop(X, weights, biases, num_layers)
    loss = tf.reduce_mean(tf.square(Y-Yhat))
    train = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        Training_loss = []
        for n in range(int(Nlearning*n_copy/n_batch)):
            input_X = np.reshape(trainX_total[n*n_batch:(n+1)*n_batch], [n_batch, INPUT_SIZE])
            output_Y = np.reshape(trainY_total[n*n_batch:(n+1)*n_batch], [n_batch, OUTPUT_SIZE])
            feed = {X: input_X, Y: output_Y}
            _, temp = sess.run([train, loss], feed_dict=feed)
            Training_loss.append(temp)

        # Save
        save_weights(weights, biases, output_folder, weight_name_save, num_layers)

        # Test
        Test_loss = []
        test_batch = int(n_batch/20)
        for n in range(int(Ntest/test_batch)):
            input_X = np.reshape(testX[n*test_batch:(n+1)*test_batch], [test_batch, INPUT_SIZE])
            output_Y = np.reshape(testY[n*test_batch:(n+1)*test_batch], [test_batch, OUTPUT_SIZE])
            feed = {X: input_X, Y: output_Y}
            _, temp = sess.run([train, loss], feed_dict=feed)
            Test_loss.append(temp)

        # Example test
        Tstate = np.random.randint(2, size=N_pixel)
        TR = pixelDBR.calR(Tstate, dx, N_pixel, wavelength, nh, nl)
        tX = np.reshape(Tstate, [-1, INPUT_SIZE])
        tY = np.reshape(TR, [-1, OUTPUT_SIZE])
        NR = sess.run(Yhat, feed_dict={X: tX})
        Tloss = sess.run(loss, feed_dict={X: tX, Y: tY})
    # sess.close()

    print("LOSS: ", Tloss)
    x = np.reshape(wavelength, wavelength.shape[1])
    TR = np.reshape(TR, wavelength.shape[1])
    NR = np.reshape(NR, wavelength.shape[1])
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(x, TR)

    plt.subplot(2, 1, 2)
    plt.plot(x, NR)

    plt.figure(3)
    plt.plot(Training_loss)
    with open('D:/1D_DBR/FCNN_1THz/Training_loss.csv', 'w') as lossfile:
        np.savetxt(lossfile, Training_loss, delimiter=',', fmt='%.5f')

    plt.figure(4)
    plt.plot(Test_loss)
    with open('D:/1D_DBR/FCNN_1THz/Test_loss.csv', 'w') as lossfile:
        np.savetxt(lossfile, Test_loss, delimiter=',', fmt='%.5f')
    plt.show()



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--output_folder",type=str, default='D:/1D_DBR/FCNN_1THz/NN_parameter')
    parser.add_argument("--weight_name_save", type=str, default="")
    parser.add_argument("--n_batch", type=int, default=100)
    parser.add_argument("--lr_rate", type=float, default=1E-3)
    parser.add_argument("--num_layers", default=3)
    parser.add_argument("--n_hidden", default=200)

    args = parser.parse_args()
    dict = vars(args)

    for key, value in dict.items():
        if (dict[key]=="False"):
            dict[key] = False
        elif dict[key]=="True":
            dict[key] = True
        try:
            if dict[key].is_integer():
                dict[key] = int(dict[key])
            else:
                dict[key] = float(dict[key])
        except:
            pass
    print(dict)

    kwargs = {  
            'output_folder':dict['output_folder'],
            'weight_name_save':dict['weight_name_save'],
            'n_batch':dict['n_batch'],
            'lr_rate':dict['lr_rate'],
            'num_layers':int(dict['num_layers']),
            'n_hidden':int(dict['n_hidden'])
            }

    main(**kwargs)
    # Ratio_Optimization(**kwargs)