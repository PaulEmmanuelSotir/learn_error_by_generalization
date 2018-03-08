#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Generalization loss """
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10, cifar100

import utils
import dnn

__all__ = ['build_graph', 'train', 'load_cifar']

# Hyperparameters (argparse configuration)
hyperparameters_config = {
    '--lr': {'default': 1e-4, 'metavar': '-l', 'type': float, 'help': 'learning rate'},
    '--l2_reg': {'default': 5e-4, 'metavar': '-r', 'type': float, 'help': 'L2 regularization factor'},
    '--epochs': {'default': 100, 'metavar': '-e', 'type': int, 'help': 'Number of training epochs'},
    '--dropout': {'default': 1., 'metavar': '-d', 'type': float, 'help': 'Dropout regularization'},
    '--momentum': {'default': 0.9, 'metavar': '-m', 'type': float, 'help': 'SGD Momentum'},
    '--batch_size': {'default': 64, 'metavar': '-b', 'type': int, 'help': 'Batch size to train on'},
    '--batch_norm': {'default': True, 'metavar': '-n', 'type': bool, 'help': 'Enable or disable batch normalization'},
    '--weight_decay': {'default': 5e-4, 'type': float, 'help': 'Weight decay (L2 regularization)'},
    '--layers': {'default': [256] * 3, 'type': int, 'help': 'Fully connected hidden layer sizes', 'nargs': '+'},
    '--save_dir': {'default': '/home/pes/deeplearning/models/generalization_training/train_1/', 'type': str, 'help': 'Tensorflow model save directory'}
}

INFERENCE_BATCH_SIZE = 1024
ALLOW_GPU_MEM_GROWTH = True
USE_CIFAR100 = False
INPUT_SIZE = 32*32*3
N_CLASSES = 100 if USE_CIFAR100 else 10


def main():
    # Parse cmd arguments
    hp = utils.hyperparameters_from_args(hyperparameters_config, description='Fully connected neural network training on CIFAR-100')
    dataset = load_cifar()
    ops = build_graph(hp)
    train(hp, dataset, ops)


def train(hp, dataset, ops):
    (train_x, train_y), (test_x, test_y) = dataset
    (model, trainer, saver, init_ops) = ops

    with tf.Session(config=utils.tf_config(ALLOW_GPU_MEM_GROWTH)) as sess:
        # Initialize parameters and create summary writer
        best_acc = 0.
        sess.run(init_ops)
        shutil.rmtree(hp['save_dir'], ignore_errors=True)
        os.makedirs(hp['save_dir'], exist_ok=True)
        summary_writer = tf.summary.FileWriter(hp['save_dir'], sess.graph)

        for epoch in range(hp['epochs']):
            print('\n' + '-' * 80 + '\nEpoch %03d/%03d' % (epoch + 1, hp['epochs']))
            # Train and evaluate model
            train_loss = _train_epoch(trainer, sess, train_x, train_y, hp)
            validloss, validacc = _valid(trainer, sess, test_x, test_y, hp)
            print('\ttrain_loss=%2.5f\tvalid_loss=%2.5f\tvalid_acc=%3.4f' % (train_loss, validloss, validacc))
            # Summaries metrics
            utils.add_summary_values(summary_writer, global_step=epoch, train_loss=train_loss, validloss=validloss, validacc=validacc)
            # Save model if accuracy improvement
            if validacc > best_acc:
                print('\tBest accuracy encountered so far, saving model...')
                best_acc = validacc
                saver.save(sess, hp['save_dir'])
    return best_acc


def build_graph(hp):
    # Define and train FC neural network
    model = dnn.DNN(INPUT_SIZE, N_CLASSES, hp)
    trainer = dnn.Trainer(hp, model)
    saver = tf.train.Saver()
    init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name='init_op')
    return model, trainer, saver, init_ops


def load_cifar():
    # Load CIFAR-100 dataset and normalize it over each channels
    def _normalize(samples):
        v_min = samples.min(axis=(0, 1, 2), keepdims=True)
        v_max = samples.max(axis=(0, 1, 2), keepdims=True)
        return (samples - v_min)/(v_max - v_min)
    (train_x, train_y), (test_x, test_y) = cifar100.load_data(label_mode='fine') if USE_CIFAR100 else cifar10.load_data()
    (train_x, train_y) = np.reshape(_normalize(train_x), [-1, INPUT_SIZE]), np.reshape(train_y, [-1])
    (test_x, test_y) = np.reshape(_normalize(test_x), [-1, INPUT_SIZE]), np.reshape(test_y, [-1])
    return (train_x, train_y), (test_x, test_y)


def _train_epoch(trainer, sess, train_x, train_y, hp):
    tot_loss = 0
    for step, batch_size, (batch_x, batch_y) in utils.batch(hp['batch_size'], train_x, train_y, shuffle=True):
        loss = trainer.fit(sess, batch_x, batch_y)
        tot_loss += loss * batch_size / len(train_x)
    return tot_loss


def _valid(trainer, sess, test_x, test_y, hp):
    tot_loss, tot_acc = (0, 0)
    for step, batch_size, (batch_x, batch_y) in utils.batch(hp['batch_size'], test_x, test_y):
        probs, classes, loss = trainer.evaluate(sess, batch_x, batch_y)
        tot_acc += np.sum(np.equal(classes, batch_y)) / len(test_x)
        tot_loss += loss * batch_size / len(test_x)
    return (tot_loss, tot_acc)


if __name__ == '__main__':
    main()
