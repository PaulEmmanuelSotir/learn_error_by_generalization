#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Random hyperparameter search on DNN model """
import os
import re
import time
import threading
import argparse
import numpy as np
import tensorflow as tf

from DNN import train_dnn
import learn_error_by_generalization as egt
import utils

__all__ = ['hyperparameter_search']

USE_CIFAR100 = False

DNN_HP_DOMAIN = {
    'lr': [5e-4, 0.25],
    'l2_reg': [0., 5e-2],
    'epochs': 100,
    'dropout': [0.6, 1.],
    'momentum': [0.65, 0.90],
    'batch_size': (64, 128, 256, 512),
    'batch_norm': (True, False),
    'layers': [256] * 4,
    'extended_summary': False
}

EGT_HP_DOMAIN = {
    'lr': [1e-4, 1e-1],
    'steps': 32,
    'batch_size': (32, 64, 128, 256),
    'sub_lr': 0.004810949156452254,  # TODO: make hyperparameter search over this later
    'sub_steps': (64, 128, 256, 512),
    'sub_momentum': 0.7535053776517011,  # TODO: make hyperparameter search over this later
    'sub_batch_size': (64, 128, 256, 512),
    'trained_model_dir': '/home/pes/deeplearning/models/cifar10_dnn/train_dnn_1/',
    'dnn_hp': {'lr': 0.004810949156452254,
               'l2_reg': 0.004088853747157203,
               'epochs': 40,
               'dropout': 1.0,
               'momentum': 0.7535053776517011,
               'batch_size': 128,
               'batch_norm': False,
               'layers': [256] * 6}
}


def hyperparameter_search(hp_domain, n_iter, n_replicas, save_dir, model):
    """ Random hyperparameter search on DNN model """
    dataset = utils.load_cifar(USE_CIFAR100)
    start_time = time.time()
    gpu_devices = utils.get_available_gpus()

    threads = []
    for rep_id in range(n_replicas):
        threads.append(threading.Thread(target=_hp_search_replica, kwargs={'start_time': start_time, 'gpu_devices': gpu_devices, 'hp_domain': hp_domain,
                                                                           'save_dir': save_dir, 'dataset': dataset, 'rep_id': rep_id, 'n_iter': n_iter, 'model': model}))
        threads[-1].start()

    for thread in threads:
        thread.join()

    # Look for best hyperparameter set we found
    best_results = {'acc': float('-inf'), 'model_name': None, 'hp': None}
    for root, dirs, files in os.walk(save_dir):
        for d in dirs:
            if re.match('model_([0-9]+)_([0-9]+)', d) is not None:
                results = np.load(os.path.join(root, d, 'results.npy')).tolist()
                if best_results['acc'] < results['acc']:
                    best_results = results
                    best_results['model_name'] = d

    print('\n\nHyperparameter search done!\n\tbest_acc=%4f\n\tbest_model_name=%s' % (best_results['acc'], best_results['model_name']) + '\n' + '#' * 100)
    return best_results


def _hp_search_replica(start_time, gpu_devices, rep_id, n_iter, hp_domain, save_dir, dataset, model):
    for i in range(n_iter):
        model_name = 'model_%d_%d' % (rep_id, i)
        elapsed_time = time.time() - start_time
        print('\n' * 4 + '#' * 100 + '\n>\tHyperparameter_set#%d of replica %d, elapsed_time=%ds\n>\tTraining on a new hyperparameter set:' % (i, rep_id, elapsed_time))

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
        hp['save_dir'] = os.path.join(save_dir, model_name, '')

        # Build and train model
        g = tf.Graph()
        with g.as_default():
            with tf.device(gpu_devices[rep_id % len(gpu_devices)]):
                print(hp)
                ops = model.build_graph(hp)
            acc = model.train(hp, dataset, ops)
        np.save(os.path.join(hp['save_dir'], 'results.npy'), {'hp': hp, 'acc': acc})
        print('\n>\taccuracy=%4f' % (acc) + '\n' + '#' * 100)


if __name__ == '__main__':
    # Parse CMD args
    parser = argparse.ArgumentParser(description="Hyperparameter search on CIFAR dataset")
    parser.add_argument('--save_dir', metavar='-s', type=str, help='Directory where tensorflow models and hyperparameter search results will be saved')
    parser.add_argument('--n_replicas', metavar='-p', type=int, default=1, help='Number of tensorflow graphs replicas to execute in parallel')
    parser.add_argument('--n_iter', metavar='-i', type=int, default=128, help='Number of hyperparameter set samples tested')
    parser.add_argument('--model', metavar='-m', type=str, choices=['DNN', 'EGT'], default='DNN', help="Either 'DNN' or 'EGT'")
    args = parser.parse_args()
    hp_domain = DNN_HP_DOMAIN if args.model == 'DNN' else EGT_HP_DOMAIN
    save_dir = utils.replace_dir(args.save_dir)

    # Run hyperparameter search
    best_results = hyperparameter_search(hp_domain, args.n_iter, args.n_replicas, save_dir, train_dnn if args.model == 'DNN' else egt)
    np.save(os.path.join(save_dir, 'best_hp.npy'), best_results)
