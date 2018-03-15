#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Simple fully connected neural network """
import tensorflow as tf
import numpy as np

import utils


class Trainer(object):
    def __init__(self, hp, model):
        self.lr = hp['lr']
        self.model = model
        self.l2_reg = hp['l2_reg']
        self.momentum = hp['momentum']

        # Define loss and optimizer
        with tf.variable_scope('L2_regularization'):
            L2 = self.l2_reg * tf.add_n([tf.nn.l2_loss(w) for w in self.model.weights])
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.model.y, logits=self.model.logits), name='xentropy') + L2
        self.optimize = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum, use_nesterov=True).minimize(self.loss)

    def fit(self, sess, X, y):
        """ Train model on minibatch data """
        _, loss = sess.run((self.optimize, self.loss), feed_dict={self.model.x: X, self.model.y: y, self.model.training: True})
        return loss

    def predict(self, sess, X):
        """ Infer class probabilities from given input """
        probs = sess.run(self.model.probs, feed_dict={self.model.x: X})
        return probs, np.argmax(probs, axis=1)

    def evaluate(self, sess, X, y):
        """ Evaluate model on given input and labels """
        probs, loss = sess.run((self.model.probs, self.loss), feed_dict={self.model.x: X, self.model.y: y})
        return probs, np.argmax(probs, axis=1), loss


class DNN(object):
    """ Fully connected neural network model with batch normalization and dropout """

    def __init__(self, input_size, n_classes, hp, act_fn=tf.nn.relu6, weight_init=utils.relu_xavier_avg):
        # Hyperparameters
        self.n_classes = n_classes
        self.weight_init = weight_init
        self.activation_fn = act_fn
        self.input_size = input_size
        self.layers = hp['layers']
        self.dropout = hp['dropout']
        self.batch_norm = hp['batch_norm']

        # Model definition
        self.weights, self.biases = [], []
        self._build_model()

    def _build_model(self):
        with tf.variable_scope('DNN'):
            # Define input placeholders
            self.y = tf.placeholder(tf.int32, [None], name='y')
            self.x = tf.placeholder(tf.float32, [None, self.input_size], name='X')
            self.training = tf.placeholder_with_default(False, [], name='training')  # Needed for batch normalization
            # Fully connected NN layers
            layer = self._dense_layer(self.x, [self.input_size, self.layers[0]], 'input_layer',
                                      False, self.weight_init, self.activation_fn, self.dropout)
            for i, in_size, out_size in zip(range(1, len(self.layers)), self.layers[:-1], self.layers[1:]):
                layer = self._dense_layer(layer, [in_size, out_size], 'layer_{}'.format(i), self.batch_norm, self.weight_init, self.activation_fn, self.dropout)
            self.logits = self._dense_layer(layer, [self.layers[-1], self.n_classes], 'output_layer', self.batch_norm, self.weight_init)
            with tf.variable_scope('output_layer'):
                self.probs = tf.nn.softmax(self.logits, name='probs')

    def _dense_layer(self, x, shape, name, batch_norm, init, activation_fn=None, keep_prob=1.):
        with tf.variable_scope(name):
            self.weights.append(tf.get_variable(initializer=init(shape), name='w'))
            self.biases.append(tf.get_variable(initializer=tf.truncated_normal([shape[1]]) if shape[1] > 1 else 0., name='b'))
            logits = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])
            if batch_norm:
                logits = tf.layers.batch_normalization(logits, training=self.training, name='batch_norm')
            dense = activation_fn(logits) if activation_fn is not None else logits
            dense = tf.nn.dropout(dense, keep_prob, name='logits')
        return dense
