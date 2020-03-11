import numpy as np
from PIL import Image
import pdb


def read_data(filepath, type_="train", class_num=None):
    """Read different types of data from different files."""
    if type_ == "train":
        return read_train(filepath)
    elif type_ == "test":
        return read_test(filepath)
    elif type_ == "ground" and class_num is not None:
        return read_ground_truth(filepath, class_num)
    else:
        raise ValueError("Invalid input for the file type")


def read_train(filepath):
    """Read data from train file."""
    f = open(filepath)
    X = []
    Y = []
    class_num = {}
    class_counter = 0
    for line in f:
        data = line.strip().split(" ")
        img = Image.open(data[0]).convert("L")
        img.thumbnail((64, 64), Image.ANTIALIAS)
        img = np.array(img)
        X.append(img.flatten())
        if data[1] not in class_num:
            class_num[data[1]] = class_counter
            class_counter += 1
        Y.append(class_num[data[1]])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, class_num


def read_test(filepath):
    """Read sample from test file."""
    f = open(filepath)
    X = []
    for line in f:
        data = line.strip()
        img = Image.open(data).convert("L")
        img.thumbnail((64, 64), Image.ANTIALIAS)
        img = np.array(img)
        X.append(img.flatten())
    X = np.array(X)
    return X


def read_ground_truth(filepath, class_num):
    """Read ground truth values, same format as output."""
    f = open(filepath)
    Y = []
    for line in f:
        data = line.strip()
        Y.append(class_num[data])
    Y = np.array(Y)
    return Y
