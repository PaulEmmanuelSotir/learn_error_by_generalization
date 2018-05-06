#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Random hyperparameter search on DNN model """
import os
import time
import threading
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


def hyperparameter_search(hp_domain, n_iter, n_replicas):
    """ Random hyperparameter search on DNN model """
    dataset = utils.load_cifar(USE_CIFAR100)
    mutex = threading.Semaphore()
    best_results = {'acc': 0, 'model_name': None, 'hp': None}
    start_time = time.time()
    gpu_devices = utils.get_available_gpus()

    threads = []
    for rep_id in range(n_replicas):
        threads.append(threading.Thread(target=_hp_search_replica, kwargs={'start_time': start_time, 'gpu_devices': gpu_devices, 'hp_domain': hp_domain, 'mutex': mutex,
                                                                           'dataset': dataset, 'best_results': best_results, 'rep_id': rep_id, 'n_iter': n_iter}))
        threads[-1].start()

    for thread in threads:
        thread.join()

    print('\n\nHyperparameter search done!\n\tbest_acc=%4f\n\tbest_model_name=%s' % (best_results['acc'], best_results['model_name']) + '\n' + '#' * 100)
    return best_results


def _hp_search_replica(start_time, gpu_devices, mutex, best_results, rep_id, n_iter, hp_domain, dataset):
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
        hp['save_dir'] = os.path.join(hp['save_dir'], model_name)

        # Build and train model
        g = tf.Graph()
        with g.as_default():
            with tf.device(gpu_devices[rep_id % len(gpu_devices)]):
                ops = build_graph(hp)
            acc = train(hp, dataset, ops)

        if acc > best_results['acc']:
            print('\n>\tBest hyperparameter set encountered so far!')
            mutex.acquire()
            best_results['hp'] = hp
            best_results['acc'] = acc
            best_results['model_name'] = model_name
            mutex.release()
        np.save(os.path.join(hp['save_dir'], 'results.npy'), {'hp': hp, 'acc': acc})
        print('\n>\taccuracy=%4f' % (acc) + '\n' + '#' * 100)


if __name__ == '__main__':
    # Parse CMD args
    parser = argparse.ArgumentParser(description="Hyperparameter search for FC neural net trained on CIFAR dataset")
    parser.add_argument('--save_dir', metavar='-s', type=str, default=HP_DOMAIN['save_dir'],
                        help='Directory where tensorflow models and hyperparameter search results will be saved')
    parser.add_argument('--n_replicas', metavar='-p', type=int, default=1, help='Number of tensorflow graphs replicas to execute in parallel')
    args = parser.parse_args()
    HP_DOMAIN['save_dir'] = args.save_dir
    HP_DOMAIN['save_dir'] = os.path.join(HP_DOMAIN['save_dir'], '')

    # Run hyperparameter search
    best_results = hyperparameter_search(HP_DOMAIN, n_iter=N_ITER, n_replicas=args.n_replicas)
    np.save(os.path.join(HP_DOMAIN['save_dir'], 'best_hp.npy'), best_results)
