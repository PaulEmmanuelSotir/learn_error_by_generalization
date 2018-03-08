""" Random hyperparameter search on DNN model """
import os
import time
import numpy as np
import tensorflow as tf

from main import build_graph, train, load_cifar

N_ITER = 200
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
    'save_dir': '/home/pes/deeplearning/models/generalization_training/hyperparameter_search/'
}


def main(hp_domain, n_iter):
    """ Random hyperparameter search on DNN model """
    dataset = load_cifar()
    best_acc, best_model_name = 0., 'model_0'
    start_time = time.time()

    for i in range(n_iter):
        elapsed_time = time.time() - start_time
        print('\n' * 4 + '#' * 100 + '\n>\tHyperparameter_set#%d, elapsed_time=%2f\n>\tTraining on a new hyperparameter set:' % (i, elapsed_time))

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
        print('>\taccuracy=%4f' % (acc) + '\n' + '#' * 100)

    print('\n\nHyperparameter search done!\n\tbest_acc=%4f\n\tbest_model_name=%s' % (best_acc, best_model_name) + '\n' + '#' * 100)
    return best_hp, best_acc, best_model_name


if __name__ == '__main__':
    best_hp, best_acc, best_model_name = main(HP_DOMAIN, n_iter=N_ITER)
    np.save(os.path.join(HP_DOMAIN['save_dir'], 'best_hp.npy'), {'hp': best_hp, 'acc': best_acc, 'model_name': best_model_name})
