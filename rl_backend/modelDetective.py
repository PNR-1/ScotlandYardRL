import itertools
import numpy as np
import os
import random
import tensorflow as tf
import utilities
class Model(object):

    def __init__(self):
        tf.reset_default_graph
        self.learning_rate = 0.001

        self.columns = 1432
        self.X = tf.placeholder(shape=[None, self.columns], dtype=tf.float32, name="X")
    # The TD target value
        self.Y = tf.placeholder(shape=[None,1], dtype=tf.float32, name="Y")


        self.W1 = tf.Variable(tf.random_normal([self.columns, 128]))
        self.b1 = tf.Variable(tf.random_normal([128]))
        self.hidden1 = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)

        self.W2 = tf.Variable(tf.random_normal([128, 64]))
        self.b2 = tf.Variable(tf.random_normal([64]))
        self.hidden2 = tf.nn.relu(tf.matmul(self.hidden1, self.W2) + self.b2)

        self.W3 = tf.Variable(tf.random_normal([64, 1]))
        self.b3 = tf.Variable(tf.random_normal([1]))
        self.pred = tf.nn.relu(tf.matmul(self.hidden2, self.W3) + self.b3)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        #print("hello")
        self.init = tf.initialize_all_variables()
        global sess
        sess = tf.Session()
        sess.run(self.init)



    def initialize(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
        return sess


    def predict(self,x):
        y = sess.run(self.pred, feed_dict={self.X:x})
        return y


    def optimize(self,x, y):
        #sess = tf.Session()
        sess.run([self.optimizer, self.cost], feed_dict={self.X: x, self.Y: y})
