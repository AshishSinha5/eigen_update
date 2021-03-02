import os
from PIL import Image
import numpy as np
import pickle


def get_face94_male(file_path, img_per_class=2):
    data = {}
    for i, dirs in enumerate(os.listdir(file_path)):
        img_path = file_path + dirs + '/'
        for j, img in enumerate(os.listdir(img_path)):
            if '.jpg' in img:
                if j == (img_per_class):
                    break
                if i not in data:
                    data[i] = [np.array(Image.open(img_path + img).convert('L'))]
                else:
                    data[i].append(np.array(Image.open(img_path + img).convert('L')))
    return data


def get_coil100(file_path):
    data = {}
    pass


if __name__ == "__main__":
    file_path = "data/faces94/male/"
    data = get_face94_male(file_path, img_per_class=3)
    with open('data/data_dict.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
