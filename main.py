from get_eigen_space import get_eigen_space
from datagen import get_face94_male
import pickle
import os
from test import test
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', help="file path to image data", type=str, default='data/faces94/male/')
    parser.add_argument('-e', '--epsilon', help='tolerance value for svd update', type=float, default=0.3)
    parser.add_argument('-n', '--num_image', help='number of images per class', type = int, default=3)
    parser.add_argument('-ntr', '--num_train', help='number of training sample per object', type=int, default=2)

    args = parser.parse_args()
    file_path = args.file_path
    eps = args.epsilon
    num_image = args.num_image
    num_train = args.num_train

    print("Getting Data")
    data = get_face94_male(file_path, num_image)

    print("Getting eigenfaces")
    eigen = get_eigen_space(data, eps, num_train)
    U, sigma, V, wt = eigen.eigenSpaceUpdate()

    fts = ['U', 'sigma', 'V', 'wt']
    ft_dict = dict(zip(fts, [U, sigma, V, wt]))

    print('Tesing on test set')
    test = test(data, ft_dict.values(), num_train)
    acc = test.test()
    print("Test accuracy at eps {}  = {}".format(eps, acc))
    if not os.path.exists('data/face94ft'):
        os.makedirs('data/face94ft/')

    with open('data/face94ft/feat{}.pickle'.format(eps), 'wb') as f:
        pickle.dump(ft_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
