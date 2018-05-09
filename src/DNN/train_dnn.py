#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Generalization loss """
import numpy as np
import tensorflow as tf

import utils
from DNN import dnn

# TODO: try other optimization methods
__all__ = ['build_graph', 'train']

# Hyperparameters (argparse configuration)
hyperparameters_config = {
    'lr': {'default': 0.004810949156452254, 'metavar': '-l', 'type': float, 'help': 'learning rate'},
    'l2_reg': {'default': 0.004088853747157203, 'metavar': '-r', 'type': float, 'help': 'L2 regularization factor (weight decay)'},
    'epochs': {'default': 200, 'metavar': '-e', 'type': int, 'help': 'Number of training epochs'},
    'dropout': {'default': 1., 'metavar': '-d', 'type': float, 'help': 'Dropout regularization'},
    'momentum': {'default': 0.7535053776517011, 'metavar': '-m', 'type': float, 'help': 'SGD Momentum'},
    'batch_size': {'default': 128, 'metavar': '-b', 'type': int, 'help': 'Batch size to train on'},
    'batch_norm': {'default': False, 'metavar': '-n', 'type': bool, 'help': 'Enable or disable batch normalization'},
    'layers': {'default': [256] * 6, 'type': int, 'help': 'Fully connected hidden layer sizes', 'nargs': '+'},
    'save_dir': {'default': '/home/pes/deeplearning/models/cifar10_dnn/train_dnn_1/', 'type': str, 'help': 'Tensorflow model save directory'}
}

INFERENCE_BATCH_SIZE = 1024
ALLOW_GPU_MEM_GROWTH = True
USE_CIFAR100 = False
N_CLASSES = 100 if USE_CIFAR100 else 10


def main():
    # Parse cmd arguments
    hp = vars(utils.hyperparameters_from_args(hyperparameters_config, description='Fully connected neural network training on CIFAR-100'))
    dataset = utils.load_cifar(USE_CIFAR100)
    ops = build_graph(hp)
    train(hp, dataset, ops)


def train(hp, dataset, ops):
    (train_x, train_y), (test_x, test_y) = dataset
    (model, trainer, saver, init_ops) = ops

    with tf.Session(config=utils.tf_config(ALLOW_GPU_MEM_GROWTH)) as sess:
        # Initialize parameters and create summary writer
        best_acc = 0.
        sess.run(init_ops)
        hp['save_dir'] = utils.replace_dir(hp['save_dir'])
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
    model = dnn.DNN(utils.CIFAR_INPUT_SIZE, N_CLASSES, hp)
    trainer = dnn.Trainer(hp, model)
    saver = tf.train.Saver()
    init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name='init_op')
    return model, trainer, saver, init_ops


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
