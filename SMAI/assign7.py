import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Activation(W, x):
    """Activation function."""
    return np.dot(W, x)


def Perceptron(X, Y, lr=0.1, epochs=200):
    """Perceptron."""
    n, d = X.shape
    X_aug = np.hstack((X, np.ones((n, 1))))
    # Initialised with line x1 = 1
    W = np.array([1, 0, -1])

    print(W)

    weights = []

    for t in range(epochs):
        print("\n\nEpoch: {}".format(t))
        for i, x in enumerate(X_aug):
            y_pred = 1 if Activation(W, x) >= 0 else -1
            update = lr * (Y[i] - y_pred) * x
            W = W + update
            weights.append(W)
            print("x = {}, y_pred: {}, y_actual = {}, update={}".format(x, y_pred, Y[i], update))
            print("New Weight: {}".format(W))

    return W, np.array(weights)


if __name__ == "__main__":
    X = np.array([
        [1, 1],
        [-1, -1],
        [2, 2],
        [-2, -2],
        [-1, 1],
        [1, -1]
    ])

    Y = np.array([-1, -1, 1, -1, 1, 1])

    W, weights = Perceptron(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(weights[:, 0], weights[:, 1], weights[:, 2])
    fig.show()
    pdb.set_trace()
