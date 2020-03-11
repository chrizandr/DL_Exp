"""PCA."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
from data import read_data
import pdb
import pickle


def PCA(X, n_components):
    """PCA reduction, reduce to n_components"""
    try:
        f = open("pca_cov_evecs.pkl".format(), "rb")
        covariance, evals, evecs = pickle.load(f)
        f.close()
    except FileNotFoundError:
        X = X.astype(np.float)
        X -= np.mean(X, axis=0)
        covariance = np.cov(X, rowvar=False)
        evals, evecs = np.linalg.eigh(covariance)

        # Sort Eigne vectors and values
        indices = np.argsort(evals)[::-1]
        evals = evals[indices]
        evecs = evecs[:, indices]
        f = open("pca_cov_evecs.pkl", "wb")
        pickle.dump((covariance, evals, evecs), f)
        f.close()

    max_evec = evecs[:, 0:n_components]
    if evals.dtype == np.complex128:
        max_evec = max_evec.real
    X = np.dot(X, max_evec)
    return X, max_evec


def reconstruct(X_reduced, max_evec, mean):
    X = np.dot(X_reduced, max_evec.T)
    X = X + mean
    return X


def MSE(X, X_reconstructed):
    return ((X - X_reconstructed) ** 2).mean(axis=0).sum()


if __name__ == "__main__":
    data_path = "train.txt"
    X, Y, class_num = read_data(data_path, type_="train")

    sizes = list(range(8, 4097, 8))
    errors = []
    for i in sizes:
        print("Reducing and reconstructing for size {}".format(i))
        X_reduced, max_evec = PCA(X, n_components=i)
        X_reconstructed = reconstruct(X_reduced, max_evec, np.mean(X, axis=0))
        errors.append(MSE(X, X_reconstructed))
    pdb.set_trace()
    plt.plot(sizes, errors, label="Error vs N-Components to reconstruct")
    plt.xlabel("N-Components")
    plt.ylabel("Total MSE")
    plt.show()

    X_1d, _ = PCA(X, n_components=1)
    X_2d, _ = PCA(X, n_components=2)
    X_3d, _ = PCA(X, n_components=3)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink']

    for c in np.unique(Y):
        indices = (Y == c).nonzero()[0]
        X_points = X_1d[indices]
        plt.scatter(X_points[:, 0], np.zeros(X_points.shape[0]), color=colors[c], label="Class {}".format(c))
    plt.xlabel("1D point x1")
    plt.legend()
    plt.show()

    for c in np.unique(Y):
        indices = (Y == c).nonzero()[0]
        X_points = X_2d[indices]
        plt.scatter(X_points[:, 0], X_points[:, 1], color=colors[c], label="Class {}".format(c))
    plt.xlabel("2D point x1")
    plt.ylabel("2D point x2")
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c in np.unique(Y):
        indices = (Y == c).nonzero()[0]
        X_points = X_3d[indices]
        ax.scatter(X_points[:, 0], X_points[:, 1], X_points[:, 1], color=colors[c], label="Class {}".format(c))
    ax.set_xlabel("3D point x1")
    ax.set_ylabel("3D point x2")
    ax.set_zlabel("3D point x2")
    plt.legend()
    plt.show()
