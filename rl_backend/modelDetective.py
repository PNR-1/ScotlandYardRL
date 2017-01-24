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
        self.columns = 1415
        self.X = tf.placeholder(shape=[None, self.columns], dtype=tf.float32, name="X")
    # The TD target value
        self.Y = tf.placeholder(shape=[1, None], dtype=tf.float32, name="Y")

        self.hidden1 = int(self.columns/2)
        self.hidden2 = int(self.columns/2)
        self.out = 1

        self.weights = {
            'w1' : tf.Variable(tf.random_normal([self.columns, self.hidden1])),
            'w2' : tf.Variable(tf.random_normal([self.hidden1, self.hidden2])),
            'out' : tf.Variable(tf.random_normal([self.hidden2, self.out]))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.hidden1])),
            'b2': tf.Variable(tf.random_normal([self.hidden2])),
            'out': tf.Variable(tf.random_normal([self.out]))
            }


        self.layer1 = tf.nn.relu(tf.matmul(self.X, self.weights['w1']) + self.biases['b1'])
        self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.weights['w2']) + self.biases['b2'])
        self.pred = tf.nn.relu(tf.matmul(self.layer2, self.weights['out']) + self.biases['out'])

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.init = tf.global_variables_initializer()

        global sess
        sess = tf.Session()
        sess.run(self.init)


    def predict(self,x):
        y = sess.run(self.pred, feed_dict={self.X:x})
        return y


    def optimize(self,x, y):
        sess.run([self.optimizer, self.cost], feed_dict={self.X: x, self.Y: y})
