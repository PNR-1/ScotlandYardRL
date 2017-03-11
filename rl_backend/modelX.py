import numpy as np
import tensorflow as tf
class Model(object):

    def __init__(self):
        tf.reset_default_graph
        self.learning_rate = 0.001
        self.truncated_backprop_length = 5
        self.columns = 1415
        self.X = tf.placeholder(shape=[None,self.columns,1], dtype=tf.float32, name="X")
    # The TD target value
        self.Y = tf.placeholder(shape=[None,1], dtype=tf.float32, name="Y")

        self.hidden1 = int(self.columns/2)
        self.hidden2 = int(self.columns/2)
        self.hidden3 = int(self.columns/4)
        self.hidden4 = int(self.columns/4)
        self.out = 1

        self.weights = {
            'w1' : tf.Variable(tf.random_normal([5, self.out])),
            'w2' : tf.Variable(tf.random_normal([self.hidden1, self.hidden2])),
            'w3' : tf.Variable(tf.random_normal([self.hidden2, self.hidden3])),
            'w4' : tf.Variable(tf.random_normal([self.hidden3, self.hidden4])),
            'out' : tf.Variable(tf.random_normal([self.hidden4, self.out]))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.out])),
            'b2': tf.Variable(tf.random_normal([self.hidden2])),
            'b3': tf.Variable(tf.random_normal([self.hidden3])),
            'b4': tf.Variable(tf.random_normal([self.hidden4])),
            'out': tf.Variable(tf.random_normal([self.out]))
            }



        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.Y))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        ##USE THIS FOR NEW VERSIONS OF TENSORFLOW
        self.init = tf.global_variables_initializer()
        ##USE THIS FOR OLDER VERSIONS OF TENSORFLOW
        #self.init = tf.initialize_all_variables()
        self.save_path = None
        #self.init_state = tf.placeholder(dtype=tf.float32, shape=[None, self.columns])
        #self.init_state = tf.Variable(tf.zeros([3, self.columns,1]))
        #Unpacking columns
        # self.input_series = tf.split(axis= 1, num_or_size_splits=self.truncated_backprop_length, value=self.X)
        #self.labels_series = tf.unstack(value=self.Y,num=1, axis=1)
        #self.input_series = tf.reshape(self.input_series, [None])
        #Forward passes
        # self.cell = tf.contrib.rnn.BasicRNNCell(self.columns)
        self.num_hidden = 5
        self.cell = tf.contrib.rnn.LSTMCell(self.num_hidden,state_is_tuple=True)
        self.weight = tf.Variable(tf.truncated_normal([self.num_hidden, 1]))
        self.bias = tf.Variable(tf.random_normal([1]))
        self.states_series, self.current_state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.X, dtype = tf.float32)#, time_major = True)#, initial_state=self.init_state)
        # Creating a saver object with a maximum of 5 latest models to be saved. The oldest one is automatically deleted
        # self.saver = tf.train.Saver(max_to_keep = 5)
        self.states_series = tf.transpose(self.states_series, [1, 0, 2])
        self.last = tf.gather(self.states_series, int(self.states_series.get_shape()[0]) - 1)

        # self.logits_series = [tf.matmul(self.last, self.weight) + self.bias] #Broadcasted addition
        # self.predictions_series = [tf.nn.softmax(logits) for logits in self.logits_series]
        #
        # self.losses = [tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        # self.cost = tf.reduce_mean(self.losses)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        global sess
        sess = tf.Session()
        sess.run(self.init)
        no_batches = 283
        #self._current_state = np.zeros(())

        self.prediction = tf.nn.softmax(tf.matmul(self.last, self.weight) + self.bias)
        self.cross_entropy = -tf.reduce_sum(self.Y * tf.log(tf.clip_by_value(self.prediction,1e-10,1.0)))
        self.optimizer = tf.train.AdamOptimizer()
        self.minimize = self.optimizer.minimize(self.cross_entropy)
        self.mistakes = tf.not_equal(tf.argmax(self.Y, 1), tf.argmax(self.prediction, 1))
        self.error = tf.reduce_mean(tf.cast(self.mistakes, tf.float32))

    def predict(self,x):

        y = sess.run(self.last, feed_dict={self.X:x})
        return y


    def optimize(self,x, y):
        # sess.run(self.init)
        # ptr = 0
        # for i in range(no_of_batches):
        #     inp, out = x[ptr:ptr+no_batches], y[ptr:ptr+np_batches]
        #     ptr += batch_size

        #_optimizer, _cost,self._current_state=
        sess.run([self.minimize], feed_dict={self.X: x, self.Y: y})#, self.init_state=self.current_state})

    def save(self, episode):
        # Saving the file path and storing the file path. The episode which it saves is appended to the file name.
        # Will make this more organized.
        self.save_path = self.saver.save(sess, 'models/x_models/', global_step = episode)

    def restore(self):
        # Restoring the file from a previous save path.
        self.saver.restore(sess, self.save_path)
