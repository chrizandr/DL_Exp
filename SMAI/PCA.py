import numpy as np
import matplotlib.pyplot as plt
import pdb


def PCA(X):
    X -= np.mean(X, axis=0)
    covariance = np.cov(X, rowvar=False)
    evals, evecs = np.linalg.eig(covariance)

    # Sort Eigne vectors and values
    indices = np.argsort(evals)[::-1]
    evals = evals[indices]
    evecs = evecs[:, indices]

    max_evec = evecs[:, 0]
    X = np.dot(X, max_evec)
    return X, max_evec


if __name__ == "__main__":
    # Data
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [-1, 0],
        [-2, -1],
        [-3, -2],
    ], dtype=np.float)

    Y = np.array([1, 1, 1, -1, -1, -1])

    X_, max_evec = PCA(X)

    plt.scatter(X[:, 0], X[:, 1], color=['red' if y > 0 else 'blue' for y in Y])
    plt.quiver([0], [0], max_evec[0], max_evec[1], color='g', scale=1.0/20)
    plt.quiver([0], [0], -1*max_evec[0], -1*max_evec[1], color='g', scale=1.0/20)
    plt.title("Data points and the Principle component")
    plt.show()

    plt.scatter(X_, np.zeros(X_.shape), color=['red' if y > 0 else 'blue' for y in Y])
    plt.plot(X_, np.zeros(X_.shape))
    plt.title("Projection of points on Principle component")
    plt.show()
