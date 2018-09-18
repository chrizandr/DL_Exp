"""Q3."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pdb


def MSE_Wrong(data, lr=0.01):
    """MSE Classification."""
    X = data[0]
    Y = data[1]
    W = np.array([1, 2])

    weights = []
    grad = np.array([10, 10])
    t = 0

    while np.linalg.norm(lr*grad) > 0.0001:
        print("\n\nEpoch: {}".format(t))
        grad = np.zeros(2)
        for i in range(len(X)):
            delta_f = 2 * (Y[i] - np.dot(W.T, X[i])) * X[i]
            grad = grad + delta_f
        grad = grad / len(X)

        W = W - (lr * grad)
        weights.append(W)

        if t > 1000:
            return W, np.array(weights), t
        t += 1

    return W, np.array(weights), t


def error(W, X, Y):
    """Find error."""
    error = 0
    for i in range(len(X)):
        error += (Y[i] - np.dot(W.T, X[i]))**2
    return error


if __name__ == "__main__":
    # Data
    X = np.array([
        [2, 4],
        [3, 4],
        [1, 2],
        [4, 5],
        [6, 7],

        [4, 3],
        [6, 5],
        [5, 3],
        [4, 1],
        [7, 6],
    ])
    Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    Y_dash = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    plt.scatter(X[:, 0], X[:, 1], color=['red' if y > 0 else 'blue' for y in Y])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    # Training
    W, Weights, t = MSE_Wrong((X, Y))
    errors = [error(w, X, Y) for w in Weights]
    plt.plot(errors, color="red", label="Error")
    plt.legend()
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.show()
    pdb.set_trace()
