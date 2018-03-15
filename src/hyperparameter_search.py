#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Random hyperparameter search on DNN model """
import os
import time
import argparse
import numpy as np
import tensorflow as tf

from train_dnn import build_graph, train
import utils

__all__ = ['hyperparameter_search']

USE_CIFAR100 = False
N_ITER = 100
HP_DOMAIN = {
    'lr': [1e-5, 0.1],
    'l2_reg': [1e-6, 5e-2],
    'epochs': 90,
    'dropout': [0.4, 1.],
    'momentum': [0.6, 0.95],
    'batch_size': (32, 64, 128, 256, 512),
    'batch_norm': (True, False),
    'weight_decay': [0., 5e-2],
    'layers': [256] * 3,
    'save_dir': '/home/pes/deeplearning/models/generalization_training/hp_search_1/'
}


def hyperparameter_search(hp_domain, n_iter):
    """ Random hyperparameter search on DNN model """
    dataset = utils.load_cifar(USE_CIFAR100)
    best_acc, best_model_name = 0., 'model_0'
    start_time = time.time()

    for i in range(n_iter):
        elapsed_time = time.time() - start_time
        print('\n' * 4 + '#' * 100 + '\n>\tHyperparameter_set#%d, elapsed_time=%ds\n>\tTraining on a new hyperparameter set:' % (i, elapsed_time))

        # Randomly sample hyperparameters from domain
        hp = {}
        for param, domain in hp_domain.items():
            if type(domain) is tuple:
                hp[param] = np.random.choice(domain)
            elif type(domain) is list and len(domain) == 2 and np.all([type(v) in np.ScalarType for v in domain]):
                hp[param] = np.random.uniform(*domain)
            else:
                hp[param] = domain
            print('\t\thp[\'%s\'] = ' % param + str(hp[param]))
        hp['save_dir'] = os.path.join(hp['save_dir'], 'model_%d/' % i)

        # Build and train model
        tf.reset_default_graph()
        ops = build_graph(hp)
        acc = train(hp, dataset, ops)
        print()
        if acc > best_acc:
            print('>\tBest hyperparameter set encountered so far!')
            best_hp = hp
            best_acc = acc
            best_model_name = 'model_%d' % i
        np.save(os.path.join(hp['save_dir'], 'results.npy'), {'hp': hp, 'acc': acc})
        print('>\taccuracy=%4f' % (acc) + '\n' + '#' * 100)

    print('\n\nHyperparameter search done!\n\tbest_acc=%4f\n\tbest_model_name=%s' % (best_acc, best_model_name) + '\n' + '#' * 100)
    return best_hp, best_acc, best_model_name


if __name__ == '__main__':
    # Parse CMD args
    parser = argparse.ArgumentParser(description="Hyperparameter search for FC neural net trained on CIFAR dataset")
    parser.add_argument('--save_dir', metavar='-s', type=str, default=HP_DOMAIN['save_dir'],
                        help='Directory where tensorflow models and hyperparameter search results will be saved')
    HP_DOMAIN['save_dir'] = parser.parse_args().save_dir

    # Run hyperparameter search
    best_hp, best_acc, best_model_name = hyperparameter_search(HP_DOMAIN, n_iter=N_ITER)
    np.save(os.path.join(HP_DOMAIN['save_dir'], 'best_hp.npy'), {'hp': best_hp, 'acc': best_acc, 'model_name': best_model_name})
