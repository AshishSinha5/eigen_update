import os
from PIL import Image
import numpy as np


def get_face94_male(file_path, img_per_class):
    data = {}
    for i, dirs in enumerate(os.listdir(file_path)):
        img_path = file_path + dirs + '/'
        for j, img in enumerate(os.listdir(img_path)):
            if '.jpg' in img:
                if i not in data:
                    data[i] = [np.array(Image.open(img_path + img).convert('L'))]
                else:
                    data[i].append(np.array(Image.open(img_path + img).convert('L')))
                if j == img_per_class:
                    break
    return data


def get_coil100(file_path):
    data = {}
    pass
