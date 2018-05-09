#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Learn error by explicit generalization
Learn to generalize explicitly by learning error vector on testset batch
"""
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
import tensorflow.contrib.graph_editor as ge
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.datasets import cifar10, cifar100

import dnn
import utils

# TODO: pretrain model on different samples than the trainset used here
# TODO: try to apply/assign updates with a low learning rate instead of getting rid of them
ALLOW_GPU_MEM_GROWTH = True
TRAINED_MODEL_DIR = '/home/pes/deeplearning/'
USE_CIFAR100 = False
N_CLASSES = 100 if USE_CIFAR100 else 10

hyperparameters_config = {
    'lr': {'default': 3e-4, 'metavar': '-l', 'type': float, 'help': 'learning rate'},
    'steps': {'default': 16, 'metavar': '-s', 'type': int, 'help': 'Number of gradient descent steps to learn error parameter'},
    'batch_size': {'default': 64, 'metavar': '-b', 'type': int, 'help': 'Training batch size'},
    'sub_lr': {'default': 0.004810949156452254, 'type': float, 'help': 'sub-learning rate'},
    'sub_steps': {'default': 64, 'type': int, 'help': 'Number of gradient descent steps to learn error parameter'},
    'sub_momentum': {'default': 0.7535053776517011, 'type': float, 'help': 'Momentum sub-optimizer parameter'},
    'sub_batch_size': {'default': 128, 'type': int, 'help': 'Test batch size'},
    'save_dir': {'default': '/home/pes/deeplearning/generalization_training/egt_1/', 'type': str, 'help': 'Tensorflow model save directory'},
    'trained_model_dir': {'default': '/home/pes/deeplearning/models/generalization_training/train_dnn_2/', 'type': str, 'help': 'Tensorflow pretrained model directory'},
    'dnn_hp': {'default': {'lr': 0.004810949156452254,
                           'l2_reg': 0.004088853747157203,
                           'epochs': 40,
                           'dropout': 1.0,
                           'momentum': 0.7535053776517011,
                           'batch_size': 128,
                           'batch_norm': False,
                           'layers': [256, 256, 256, 256, 256, 256]}, 'help': 'Pretrained DNN hyperparameters'}
}

__all__ = ['explicit_generalization_training', 'build_update_model', 'learn_error_by_generalization', 'evaluate']


def explicit_generalization_training(hp, dataset):
    # We build tensorflow graph of loaded model with parameter update operations
    # This model's trainable variables are error values on testset batch
    hp['dnn_hp']['save_dir'] = hp['trained_model_dir']
    dnn.DNN(utils.CIFAR_INPUT_SIZE, N_CLASSES, hp['dnn_hp'])
    saver = tf.train.Saver(var_list=tf.global_variables())
    update_model = build_update_model(hp)

    with tf.Session(config=utils.tf_config(ALLOW_GPU_MEM_GROWTH)) as sess:
        # Restore pretrained DNN variables
        saver.restore(sess, hp['trained_model_dir'])

        # Train the update model to minimize generalization error by finding the error vector for which weight updates imply the best score on trainset
        learn_error_by_generalization(sess, hp, update_model, dataset)


def build_update_model(hp):
    with tf.variable_scope('DNN/EGT', reuse=tf.AUTO_REUSE):
        # Compute loss on test batch from 'learnable_error' parameter and actual prediction
        # Note that learnable_error is initialized to zero, which means the intial update is equivalent to making NN learn its own output
        g = tf.get_default_graph()
        test_X_ph = g.get_tensor_by_name("DNN/X:0")
        trained_variables = [v for v in tf.global_variables() if v.name[:4] == 'DNN/']
        learnable_error = tf.Variable(tf.zeros([hp['batch_size'], N_CLASSES], tf.float32), name='learnable_error')

        logits = g.get_tensor_by_name('DNN/output_layer/logits:0')
        probs = g.get_tensor_by_name("DNN/probs:0")
        log_probs = tf.nn.log_softmax(logits, name='log_probs')
        new_probs = tf.nn.softmax(logits + learnable_error, name='new_loss')
        test_loss = tf.reduce_sum(log_probs * new_probs, name='test_loss')

        # Build updated graph
        train_X_ph = tf.placeholder(tf.float32, test_X_ph.get_shape(), name='X')
        opt = SGD(lr=hp['sub_lr'], momentum=hp['sub_momentum'], nesterov=True)
        with tf.variable_scope('adam_updates'):
            sub_updates = opt.get_updates(test_loss, trained_variables)
        replacements = utils.extract_update_dict(sub_updates)
        replacements[test_X_ph] = train_X_ph

    utils.graph_replace(test_loss, replacements, dst_scope='EGT/UpdatedDNN/', src_scope='DNN/')

    with tf.variable_scope('EGT/'):
        # Compute loss of updated graph on train batch
        train_y_ph = tf.placeholder(tf.int32, [None], name='y')
        updated_logits = g.get_tensor_by_name("EGT/UpdatedDNN/output_layer/logits:0")
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=updated_logits, labels=train_y_ph, name='generalization_xentropy')
        generalization_loss = tf.reduce_mean(xentropy, name='generalization_loss')

        # We learn 'learnable_error' by backpropagating from 'generalization_loss' through sub-SGD update on test_loss
        # Note that this step is computationally expensive as this gradient computation needs second order derivative w.r.t model parameters
        # Thus, you either need to keep model capacity low or apply this technique on top layers of your model (TODO: implement this with hessian approximation?)
        lr = tf.constant(hp['lr'], name='lr')
        meta_gradients = tf.gradients(generalization_loss, learnable_error)
        meta_optimize = tf.assign_sub(learnable_error, lr * meta_gradients[0], name='optimize')

    init_ops = tf.variables_initializer([v for v in tf.global_variables() if 'EGT/' in v.name])
    return (test_X_ph, train_X_ph, train_y_ph), generalization_loss, test_loss, probs, new_probs, meta_optimize, init_ops


def learn_error_by_generalization(sess, hp, update_model, dataset):
    (placeholders, generalization_loss, test_loss, probs, new_probs, meta_optimize, error_init) = update_model
    (test_X_ph, train_X_ph, train_y_ph) = placeholders
    (train_X, train_y), (test_X, test_y) = dataset

    mean_test_acc, mean_new_test_acc = 0, 0
    for step, test_batch_size, (batch_x_test, batch_y_test) in utils.batch(hp['batch_size'], test_X, test_y, fixed_size_batches=True):
        print('\n' + '-' * 80 + '\nStep %03d/%03d' % (step + 1, hp['steps']))
        # Initialize error vector
        sess.run(error_init)
        # Train it
        mean_loss, mean_sub_loss = 0, 0
        for sub_step, _, (batch_x, batch_y) in utils.batch(hp['sub_batch_size'], train_X, train_y, shuffle=True, fixed_size_batches=True):
            result = sess.run([meta_optimize, generalization_loss, test_loss, probs, new_probs],
                              feed_dict={test_X_ph: batch_x_test, train_X_ph: batch_x, train_y_ph: batch_y})
            _, loss, sub_loss, test_probs, test_new_probs = result
            mean_loss += loss / (hp['sub_batch_size'] * hp['sub_steps'])
            mean_sub_loss += sub_loss / (hp['sub_batch_size'] * hp['sub_steps'])
            if sub_step >= hp['sub_steps']:
                break
        # Evaluate model accuracy on testset batch given trained error and DNN prediction
        test_acc, new_test_acc = evaluate(test_probs, test_new_probs, batch_y_test)
        print('\tgeneralization_loss=%2.5f\tsub_loss=%2.5f\ttest_acc=%3.4f\tnew_test_acc=%3.4f' % (mean_loss, mean_sub_loss, test_acc, new_test_acc))
        mean_test_acc += test_acc / hp['steps']
        mean_new_test_acc += new_test_acc / hp['steps']
        if step >= hp['steps']:
            break
    print('\nTraining of error vectors done:\tmean_test_acc=%3.4f\nmean_new_test_acc=%3.4f' % (mean_test_acc, mean_new_test_acc))


def evaluate(probs, new_probs, batch_y_test):
    test_acc = np.mean(np.equal(np.argmax(probs, axis=1), batch_y_test))
    new_test_acc = np.mean(np.equal(np.argmax(new_probs, axis=1), batch_y_test))
    return test_acc, new_test_acc


if __name__ == '__main__':
    hp = vars(utils.hyperparameters_from_args(hyperparameters_config, description='Explicit generalization training of fully connected neural network on CIFAR-100'))
    dataset = utils.load_cifar(USE_CIFAR100)
    explicit_generalization_training(hp, dataset)
