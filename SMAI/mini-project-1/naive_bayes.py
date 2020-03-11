"""Naive Bayes Classification."""
from PIL import Image
import numpy as np
import sys
import operator


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


def PCA(X, n_components):
    """PCA reduction, reduce to n_components"""
    X = X.astype(np.float)
    X -= np.mean(X, axis=0)
    covariance = np.cov(X, rowvar=False)
    evals, evecs = np.linalg.eigh(covariance)

    # Sort Eigne vectors and values
    indices = np.argsort(evals)[::-1]
    evals = evals[indices]
    evecs = evecs[:, indices]

    max_evec = evecs[:, 0:n_components]
    if evals.dtype == np.complex128:
        max_evec = max_evec.real
    X = np.dot(X, max_evec)
    return X, max_evec


def argmax(prediction):
    """Argmax."""
    return max(prediction.items(), key=operator.itemgetter(1))[0]


def Bayes_Eval(X, means, covariances, probs):
    """Prediction function."""
    addition_term = {}
    inverse_covariance = {}
    predictions = []

    for i in covariances.keys():
        term = -0.5 * np.log(np.linalg.norm(covariances[i]))
        term += np.log(probs[i])
        addition_term[i] = term
        inverse_covariance[i] = np.linalg.inv(covariances[i])

    for x in X:
        prediction = {}
        for i in means.keys():
            score = np.dot((x - means[i]).T, inverse_covariance[i])
            score = np.dot(score, x - means[i])
            score = -0.5*score
            score += addition_term[i]
            prediction[i] = score
        predictions.append(prediction)

    return predictions


def Find_Mean_Covariance(X, Y):
    """Find the mean and the covariance."""
    labels = np.unique(Y)
    means = {}
    covariances = {}
    for l in labels:
        indices = (Y == l).nonzero()[0]
        X_samples = X[indices]
        means[l] = np.mean(X_samples, axis=0)
        covariances[l] = np.cov(X_samples.T)

    return means, covariances


if __name__ == "__main__":
    assert len(sys.argv) > 1

    data_path = sys.argv[1]
    test_path = sys.argv[2]

    X, Y, class_num = read_data(data_path, type_="train")
    reverse_classnum = {class_num[k]: k for k in class_num}
    classes = np.unique(Y)

    X_test = read_data(test_path, type_="test")

    X_features, max_evec = PCA(X, n_components=32)
    X_features_test = np.dot(X_test, max_evec)

    probs = {k: 1.0/len(classes) for k in classes}
    means, covariances = Find_Mean_Covariance(X_features, Y)
    scores = Bayes_Eval(X_features_test, means, covariances, probs)
    predictions = np.array([argmax(x) for x in scores])

    for p in predictions:
        print(reverse_classnum[p])
