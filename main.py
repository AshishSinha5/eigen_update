from get_eigen_space import get_eigen_space
from datagen import get_face94_male
import pickle
import os
from test import test


if __name__ == "__main__":
    file_path = 'data/faces94/male/'
    data = get_face94_male(file_path)
    eigen = get_eigen_space(data)
    U, sigma, V, wt = eigen.eigenSpaceUpdate()
    fts = ['U', 'sigma', 'V', 'wt']
    ft_dict = dict(zip(fts, [U, sigma, V, wt]))
    test = test(data, ft_dict.values())
    acc = test.test()
    print(acc)
    if not os.path.exists('data/face94ft'):
        os.makedirs('data/face94ft/')

    with open('data/face94ft/feat.pickle', 'wb') as f:
        pickle.dump(ft_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
