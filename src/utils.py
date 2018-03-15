#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Utils """
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10, cifar100

__all__ = ['tf_config', 'num_flat_features', 'hyperparameters_from_args', 'default_hyperparameters', 'batch',
           'add_summary_values', 'relu_xavier_avg', 'tanh_xavier_avg', 'linear_xavier_avg', 'leaky_relu', 'load_cifar']

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CIFAR10_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR_INPUT_SIZE = 32*32*3


# Xavier initialization helpers
RELU_XAVIER_SCALE = 2.
TANH_XAVIER_SCALE = 4.
LINEAR_XAVIER_SCALE = 1.
relu_xavier_avg = tf.variance_scaling_initializer(RELU_XAVIER_SCALE, mode="fan_avg")
tanh_xavier_avg = tf.variance_scaling_initializer(TANH_XAVIER_SCALE, mode="fan_avg")
linear_xavier_avg = tf.variance_scaling_initializer(LINEAR_XAVIER_SCALE, mode="fan_avg")


def tf_config(allow_growth=True, **kwargs):
    config = tf.ConfigProto(**kwargs)
    config.gpu_options.allow_growth = allow_growth
    return config


def leaky_relu(x, leak=0.2, name='leaky_relu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def add_summary_values(summary_writer, global_step=None, **values):
    if len(values) > 0:
        summary = tf.Summary()
        for name, value in values.items():
            summary.value.add(tag=name, simple_value=value)
        summary_writer.add_summary(summary, global_step=global_step)


def hyperparameters_from_args(args_config=dict(), description=''):
    """ Parses cmd args using argparse using options specified in args_config dict """
    parser = argparse.ArgumentParser(description=description)
    for key, value in args_config.items():
        if not isinstance(value, dict):
            parser.add_argument(key, default=value)
        else:
            parser.add_argument(key, **value)
    return parser.parse_args()


def default_hyperparameters(args_config=dict()):
    hp = {}
    for key, value in args_config.items():
        key = key.lstrip('-')
        if not isinstance(value, dict):
            hp[key] = value
        elif 'default' in value:
            hp[key] = value['default']
    return hp


def num_flat_features(x, ignore_first_dim=True):
    shape = x.size()[1:] if ignore_first_dim else x.size()
    return np.prod([dim for dim in shape])


def batch(batch_size, *arrays, shuffle=False):
    """ Yields step number, effective batch size and arrays batches """
    length = len(arrays[0])
    if shuffle:
        perm = np.random.permutation(length)
        arrays = [array[perm] for array in arrays]
    batch_per_epoch = int(np.ceil(length / batch_size))
    for step, range_min in zip(range(batch_per_epoch), range(0, length - 1, batch_size)):
        range_max = min(range_min + batch_size, length)
        yield step, range_max - range_min, (array[range_min:range_max] for array in arrays)


def load_cifar(load_cifar100=False):
    # Load CIFAR-100 dataset and normalize it over each channels
    def _normalize(samples):
        v_min = samples.min(axis=(0, 1, 2), keepdims=True)
        v_max = samples.max(axis=(0, 1, 2), keepdims=True)
        return (samples - v_min)/(v_max - v_min)
    (train_x, train_y), (test_x, test_y) = cifar100.load_data(label_mode='fine') if load_cifar100 else cifar10.load_data()
    (train_x, train_y) = np.reshape(_normalize(train_x), [-1, CIFAR_INPUT_SIZE]), np.reshape(train_y, [-1])
    (test_x, test_y) = np.reshape(_normalize(test_x), [-1, CIFAR_INPUT_SIZE]), np.reshape(test_y, [-1])
    return (train_x, train_y), (test_x, test_y)
