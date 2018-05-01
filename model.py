import numpy as np
import tensorflow as tf
import pickle


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("weights", shape,initializer=initializer, dtype=tf.float32)

def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

class CNNModel(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.y_conv = 0
        self.h_conv2 = 0
        self.h_conv1 = 0
        self.h_conv5 = 0
        self.W_fc1 = 0
        self.b_fc1 = 0
        self.h_pool5 = 0

        # conv1
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([7, 7, 3, 8])
            b_conv1 = bias_variable([8])
            h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

        # conv2
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 8, 16])
            b_conv2 = bias_variable([16])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

        # conv3
        with tf.variable_scope('conv3'):
            W_conv3 = weight_variable([3, 3, 16, 32])
            b_conv3 = bias_variable([32])
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)

        # conv4
        with tf.variable_scope('conv4'):
            W_conv4 = weight_variable([5, 5, 32, 64])
            b_conv4 = bias_variable([64])
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4)

        # conv5
        with tf.variable_scope('conv5'):
            W_conv5 = weight_variable([5, 5, 64, 128])
            b_conv5 = bias_variable([128])
            h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
            h_pool5 = max_pool_2x2(h_conv5)

        # fc1
        with tf.variable_scope("fc1"):
            shape = int(np.prod(h_pool5.get_shape()[1:]))
            W_fc1 = weight_variable([shape, 1024])
            b_fc1 = bias_variable([1024])
            h_pool5_flat = tf.reshape(h_pool5, [-1, shape])
            h_fc1 = tf.matmul(h_pool5_flat, W_fc1) + b_fc1

        # dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # fc2
        with tf.variable_scope("fc2"):
            W_fc2 = weight_variable([1024, 128])
            b_fc2 = bias_variable([128])
            h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # dropout
        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

        # fc3
        with tf.variable_scope("fc3"):
            W_fc3 = weight_variable([128, 2])
            b_fc3 = bias_variable([2])
            y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3


        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv))
        self.pred = tf.argmax(y_conv, 1)

        self.correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
      
