import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

PATH = 'D:/1D_DBR/Classification'
TRAIN_PATH = 'D:/1D_DBR/trainset/04'
Nfile = 1


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


def shuffle_data(X, Y):
    Nsample = Y.shape[0]
    x = np.arange(Y.shape[0])
    np.random.shuffle(x)

    X = X[x, :]
    Y = Y[x, :]

    return X, Y


def main(n_batch, lr_rate, beta1, beta2, n_hidden):

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
    n_copy = 30
    for i in range(n_copy):
        trainX, trainY = shuffle_data(trainX, trainY)
        trainX_total = np.concatenate((trainX_total, trainX), axis=0)
        trainY_total = np.concatenate((trainY_total, trainY), axis=0)

    with tf.device('/device:GPU:0'):
        # build network
        X = tf.placeholder(tf.float32, [None, INPUT_SIZE])
        Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

        net = tf.layers.dense(Y, n_hidden[0], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="transe")
        for i, n in enumerate(n_hidden):
            shortcut = net
            net = tf.layers.dense(net, n, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense"+str(i)+"_1")
            net = tf.layers.dense(net, n, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense"+str(i)+"_2")
            net = tf.nn.relu(tf.add(net, shortcut))

        with tf.name_scope('Xhat'):
            net = tf.layers.dense(net, INPUT_SIZE, activation=tf.nn.sigmoid, name="Xhat")
        Xhat = net

        # loss = -tf.reduce_mean(X * tf.log(Xhat) + (1 - X) * tf.log(1 - Xhat))
        loss = tf.losses.log_loss(labels=X, predictions=Xhat)
        train = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(loss)
    loss_hist = tf.summary.scalar('loss', loss)

    with tf.Session() as sess:
        # tensorboard
        net.writer = tf.summary.FileWriter(PATH + '/logs/Inverse_ResNet_'+datetime.now().strftime("%Y%m%d%H%M"))
        net.writer.add_graph(sess.graph)
        # Initializer Tensorflow Variables
        sess.run(tf.global_variables_initializer())
        # Train
        for n in range(int(Nlearning*n_copy/n_batch)):
            feed_trainX = np.reshape(
                trainX_total[n*n_batch:(n+1)*n_batch, :], [n_batch, INPUT_SIZE])
            feed_trainY = np.reshape(
                trainY_total[n*n_batch:(n+1)*n_batch, :], [n_batch, OUTPUT_SIZE])
            feed_train = {X: feed_trainX, Y: feed_trainY}
            sess.run(train, feed_dict=feed_train)
            # log
            if n % 10 == 0:
                merged_summary = tf.summary.merge([loss_hist])
                summary = sess.run(merged_summary, feed_dict=feed_train)
                net.writer.add_summary(summary, global_step=n)
                print(n, 'trained')

        # Test
        test_loss = []
        for n in range(int(Ntest/n_batch)):
            feed_testX = np.reshape(
                testX[n*n_batch:(n+1)*n_batch, :], [n_batch, INPUT_SIZE])
            feed_testY = np.reshape(
                testY[n*n_batch:(n+1)*n_batch, :], [n_batch, OUTPUT_SIZE])
            feed_test = {X: feed_testX, Y: feed_testY}
            test_loss.append(sess.run(loss, feed_dict=feed_test))
        Xtest = np.reshape(sess.run(Xhat, feed_dict={Y: np.reshape(testY[99, :], [1, OUTPUT_SIZE])}), newshape=(1,INPUT_SIZE))
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.imshow(Xtest, cmap='gray', aspect='auto')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(np.reshape(testX[99, :], newshape=(1, INPUT_SIZE)), cmap='gray', aspect='auto')
        plt.colorbar()
        print(np.mean(test_loss))
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--n_batch", type=int, default=100)
    parser.add_argument("--lr_rate", type=float, default=1E-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--n_hidden", default=[600, 600, 600, 600])
    args = parser.parse_args()
    dict = vars(args)

    for key, value in dict.items():
        if (dict[key] == "False"):
            dict[key] = False
        elif dict[key] == "True":
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
            'n_batch': dict['n_batch'],
            'lr_rate': dict['lr_rate'],
            'beta1': dict['beta1'],
            'beta2': dict['beta2'],
            'n_hidden': dict['n_hidden']
            }

    main(**kwargs)