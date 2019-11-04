import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

def binaryRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    ref: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)

#As per Xaiver init, this should be 2/n(input), though many different initializations can be tried. 
def init_weights(shape,stddev=.1):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(weights)

def init_bias(shape, stddev=.1):
    """ Bias initialization """
    biases = tf.random_normal([shape], stddev=stddev)
    return tf.Variable(biases)

def save_weights(weights,biases,output_folder,weight_name_save,num_layers):
    for i in range(0, num_layers+1):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+weight_name_save+"/w_"+str(i)+".txt",weight_i,delimiter=',')
        bias_i = biases[i].eval()
        np.savetxt(output_folder+weight_name_save+"/b_"+str(i)+".txt",bias_i,delimiter=',')
    return

def load_weights(output_folder,weight_load_name,num_layers):
    weights = []
    biases = []
    for i in range(0, num_layers+1):
        weight_i = np.loadtxt(output_folder+weight_load_name+"/w_"+str(i)+".txt",delimiter=',')
        w_i = tf.Variable(weight_i,dtype=tf.float32)
        weights.append(w_i)
        bias_i = np.loadtxt(output_folder+weight_load_name+"/b_"+str(i)+".txt",delimiter=',')
        b_i = tf.Variable(bias_i,dtype=tf.float32)
        biases.append(b_i)
    return weights , biases

def forwardprop(X, weights, biases, num_layers,):
    for i in range(0, num_layers):
        if i ==0:
            htemp = tf.nn.sigmoid(tf.add(tf.matmul(X, weights[i]), biases[i]))
        else:
            htemp = tf.nn.sigmoid(tf.add(tf.matmul(htemp, weights[i]), biases[i]))
    yval = tf.add(tf.matmul(htemp, weights[-1]), biases[-1])
    return yval