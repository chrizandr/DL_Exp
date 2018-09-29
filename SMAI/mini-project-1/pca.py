"""PCA."""
import matplotlib.pyplot as plt
import sys
import numpy as np
from data import read_data
import pdb


def PCA(X, n_components):
    """PCA reduction, reduce to n_components"""
    X = X.astype(np.float)
    X -= np.mean(X, axis=0)
    covariance = np.cov(X, rowvar=False)
    evals, evecs = np.linalg.eig(covariance)

    # Sort Eigne vectors and values
    indices = np.argsort(evals)[::-1]
    evals = evals[indices]
    evecs = evecs[:, indices]

    max_evec = evecs[:, 0:n_components]
    pdb.set_trace()
    X = np.dot(X, max_evec)
    return X, max_evec


if __name__ == "__main__":
    data_path = "train.txt"
    X, Y, class_num = read_data(data_path, type_="train")
    X_reduced, max_evec = PCA(X, n_components=32)
    pdb.set_trace()
