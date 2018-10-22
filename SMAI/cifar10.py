#! /usr/bin/python2.7
import numpy as np
import cPickle as pickle
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pdb


data_dir = "data/cifar-10-batches-py/"


def load_batch(batch_num, test=False):
    if test:
        f_name = data_dir + "test_batch"
    else:
        f_name = data_dir + "data_batch_{}".format(batch_num)
    f = open(f_name, "rb")
    data = pickle.load(f)
    X = data["data"]
    Y = data["labels"]
    return X, Y


def load_data():
    X = np.empty((50000, 3072))
    Y = np.empty((50000), dtype=np.int16)

    for batch_num in [1, 2, 3, 4, 5]:
        x, y = load_batch(batch_num)
        i = batch_num - 1
        X[10000*i: 10000*(i+1)] = x
        Y[10000*i: 10000*(i+1)] = y

    return X, Y


def train_model(X, Y, type_="ovo"):
    indices = np.random.permutation(1000)
    X_sample = X[indices]
    Y_sample = Y[indices]
    model = SVC(C=1.0, kernel='linear', decision_function_shape=type_)
    model.fit(X_sample, Y_sample)
    return model


def test_model(model, reduce_model=None, reduce_model_2=None):
    X, Y = load_batch(batch_num=0, test=True)
    Y = np.array(Y)
    if reduce_model:
        X_ = reduce_model.transform(X)
        if reduce_model_2:
            X_ = reduce_model_2.transform(X_)
        pred = model.predict(X_)
    else:
        pred = model.predict(X)
    return (pred == Y).sum() / float(len(Y))


if __name__ == "__main__":
    X, Y = load_data()

    # Raw data
    print("Training")
    model = train_model(X, Y)
    print("Testing")
    accuracy = test_model(model)

    print(accuracy)
    # OVO: 0.2971, OVR:0.2972

    # With PCA
    print("Reducing with PCA")
    pca = PCA(n_components=100)
    X_ = pca.fit_transform(X)
    print("Training")
    model = train_model(X_, Y)
    print("Testing")
    accuracy = test_model(model, reduce_model=pca)

    print(accuracy)
    # OVO:0.2643, OVR:0.2619

    # With LDA
    print("Reducing with LDA")
    lda = LDA(n_components=9)
    X_ = lda.fit_transform(X, Y)
    print("Training")
    model = train_model(X_, Y)
    print("Testing")
    accuracy = test_model(model, reduce_model=lda)

    print(accuracy)
    # OVO:0.3551, OVR:0.3548

    # With PCA then LDA
    print("Reducing with PCA")
    print("Reducing with LDA")
    pca = PCA(n_components=100)
    lda = LDA(n_components=9)
    X_ = pca.fit_transform(X)
    X_ = lda.fit_transform(X_, Y)
    print("Training")
    model = train_model(X_, Y)
    print("Testing")
    accuracy = test_model(model, reduce_model=pca, reduce_model_2=lda)

    print(accuracy)
    # OVO:0.371, OVR:0.3696
