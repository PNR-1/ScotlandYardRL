import itertools
import numpy as np
import os
import random
import tensorflow as tf
import utilities


learning_rate = 0.001

columns = 0
X = tf.placeholder(shape=[None, columns], dtype=tf.float32, name="X")
    # The TD target value
Y = tf.placeholder(shape=[None,1], dtype=tf.float32, name="Y")


W1 = tf.Variable(tf.zeros([columns, 128]))
b1 = tf.Variable(tf.zeros([128]))
hidden1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.zeros([128, 64]))
b2 = tf.Variable(tf.zeros([64]))
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

W3 = tf.Variable(tf.zeros([64, 1]))
b3 = tf.Variable(tf.zeros([1]))
pred = tf.nn.relu(tf.matmul(hidden2, W3) + b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

def initialize(size):
    columns = size
    with tf.Session() as sess:
        sess.run(init)


def predict(x):
    sess = tf.session()
    y = sess.run(pred, feed_dict={X:x})
    return y


def optimize(x, y):
    sess = tf.session()
    sess.run([optimizer, cost], feed_dict={X: x, Y: y})
