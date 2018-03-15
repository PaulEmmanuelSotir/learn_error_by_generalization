#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Generalization training
Learn to generalize explicitly by learning error 
"""
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.python.keras.datasets import cifar10, cifar100

import utils

# TODO: pretrain model on different samples than the trainset used here
ALLOW_GPU_MEM_GROWTH = True
TRAINED_MODEL_DIR = '/home/pes/deeplearning/'
USE_CIFAR100 = False
N_CLASSES = 100 if USE_CIFAR100 else 10

hyperparameters_config = {
    'lr': {'default': 1e-3, 'metavar': '-l', 'type': float, 'help': 'learning rate'},
    'sub_lr': {'default': 5e-2, 'metavar': '-s', 'type': float, 'help': 'sub-learning rate'},
    'test_batch_size': {'default': 128, 'metavar': '-b', 'type': int, 'help': 'Training batch size'},
    'train_batch_size': {'default': 256, 'metavar': '-b', 'type': int, 'help': 'Test batch size'},
    'trained_model_dir': {'default': '/home/pes/deeplearning/generalization_training/train_1/', 'type': str, 'help': 'Tensorflow pretrained model directory'},
    'save_dir': {'default': '/home/pes/deeplearning/generalization_training/egt_1/', 'type': str, 'help': 'Tensorflow model save directory'}
}


def explicit_generalization_training(hp, dataset):
    # We first restore trained model
    saver = tf.train.Saver()

    with tf.Session(config=utils.tf_config(ALLOW_GPU_MEM_GROWTH)) as sess:
        saver.restore(sess, hp['trained_model_dir'])

        # We then build tensorflow graph of loaded model with parameter update opertations
        # This model's trainable variables are error values on testset batch
        update_model = build_update_model(hp)

        # Train the update model to minimize generalization error by finding the error vector for which weight updates imply the best score on trainset
        learn_error_by_generalization(sess, update_model)


def build_update_model(hp):
    with tf.variable_scope('EGT'):
        # TODO: support for batch norm parameters
        # Compute loss on test batch from 'learnable_error' parameter and actual prediction
        # Note that learnable_error is initialized to zero, which means the intial update is equivalent to making NN learn its own output
        test_X = tf.get_default_graph().get_tensor_by_name("DNN/X")
        learnable_error = tf.get_variable('learnable_error', initializer=tf.zeros_initializer(), shape=[None])
        trained_variables = [v for v in tf.global_variables() if v.name[-1] == 'w' or v.name[-1] == 'b']
        logits = tf.get_default_graph().get_tensor_by_name("DNN/logits")
        log_probs = tf.nn.log_softmax(logits, name='log_probs')
        test_loss = tf.reduce_sum(log_probs * tf.nn.softmax(logits + learnable_error))

        # Build updated graph
        replacements = {}
        for sub_grad,  w in zip(tf.gradients(test_loss, trained_variables), trained_variables):
            replacements[w] = w - hp['sub_lr'] * sub_grad
        ge.graph_replace(test_loss, replacements, dst_scope='EGT/UpdatedDNN', src_scope='DNN')

        # Compute loss of updated graph on train batch
        train_x = tf.get_default_graph().get_tensor_by_name("EGT/UpdatedDNN/X")
        train_y = tf.get_default_graph().get_tensor_by_name("EGT/UpdatedDNN/y")
        Updated_logits = tf.get_default_graph().get_tensor_by_name("EGT/UpdatedDNN/logits")
        generalization_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Updated_logits, labels=train_y), name='generalization_loss')

        # We learn 'learnable_error' by backpropagating from 'generalization_loss' through sub-SGD update on test_loss
        # Note that this step is computationally expensive as this gradient computation needs second order derivative w.r.t model parameters
        # Thus, you either need to keep model capacity low or apply this technique on top layers of your model (TODO: implement this with hessian approximation?)
        optimize = tf.group(*[tf.assign_sub(e, hp['lr'] * grad)
                              for grad, e in zip(tf.gradients(generalization_loss, learnable_error), learnable_error)], name='optimize')

    return (test_X, train_x, train_y), generalization_loss, optimize


def learn_error_by_generalization(session, update_model):
    # TODO: training loop...
    pass


if __name__ == '__main__':
    hp = utils.hyperparameters_from_args(hyperparameters_config, description='Explicit generalization training of fully connected neural network on CIFAR-100')
    dataset = utils.load_cifar(USE_CIFAR100)
    explicit_generalization_training(hp, dataset)
