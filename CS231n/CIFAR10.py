import cPickle
import numpy as np
import pdb
'''
    Function used to unpack the CIFAR-10 dataset
    Parameters : file - full path of the CIFAR-10 data set extracted file (http://www.cs.toronto.edu/~kriz/cifar.html)
    Returns a four numpy arrays
    Xtr , Ytr , Xte , Yte
    Xtr - training data ; Ytr - training data labels
    Xtr - testing data ; Ytr - testing data labels
'''
def load_Cifar(path):
    batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    test_batch = "test_batch"
    Xtr = np.array([])
    Ytr = np.array([])
    Xte, Yte = unpickle(path + test_batch)
    for batch in batches:
        data , labels = unpickle(path + batch)
        Xtr = np.vstack([Xtr, data]) if Xtr.size else data
        Ytr = np.vstack([Ytr, labels]) if Ytr.size else labels
    return Xtr, Ytr, Xte, Yte

def unpickle(file):
    fo = open(file, 'rb')
    pickle_Dict = cPickle.load(fo)
    fo.close()
    data = pickle_Dict["data"]
    labels = np.array(pickle_Dict["labels"]).reshape(-1,1)
    return data , labels


# Trying main

path = "/home/chris/Downloads/cifar-10-batches-py/"
Xtr, Ytr, Xte, Yte = load_Cifar(path)
pdb.set_trace()
