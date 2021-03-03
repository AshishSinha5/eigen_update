import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import timeit
from datagen import get_face94_male
from get_eigen_space import get_eigen_space
from test import test


def accuracy_report(file_path: str, eps: list, num_train: list):
    """
    :param file_path: file path for data
    :param eps: epsilon tolerance for SVD
    :param num_train: number of training example per class
    :return: {(eps, num_train) : (acc, time_taken, wt, U)}
    """
    rep = {}
    for e in eps:
        print("Eps = {}".format(e))
        for ntr in num_train:
            print("Training Sample per class = {}".format(ntr))
            img_per_class = ntr + 1
            data = get_face94_male(file_path, img_per_class)
            eigen = get_eigen_space(data, e, ntr)
            start = timeit.timeit()
            U, sigma, V, wt = eigen.eigenSpaceUpdate()
            time = timeit.timeit() - start
            ft_dict = dict(zip(['U', 'sigma', 'V', 'wt'], [U, sigma, V, wt]))
            testing = test(data, ft_dict.values(), ntr)
            acc = testing.test()
            rep[tuple((e, ntr))] = tuple((acc, time, wt, U))
    print('Done')
    return rep


def get_plots(rep):
    pass


if __name__ == "__main__":
    if os.path.exists('data/report.pickle'):
        f = open('data/report.pickle', 'rb')
        rep = pickle.load(f)
        get_plots(rep)
    file_path = 'data/faces94/male/'
    eps = [0.05, 0.1, 0.15, 0.2]
    n_train = [1, 2, 3]
    rep = accuracy_report(file_path, eps, n_train)
    with open('data/report.pickle', 'wb') as f:
        pickle.dump(rep, f, protocol=pickle.HIGHEST_PROTOCOL)