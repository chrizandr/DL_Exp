import numpy as np
import matplotlib.pyplot as plt
import pdb


def Activation(W, x):
    """Activation function."""
    return np.dot(W, x)


def MSE_SGD(data, lr=0.045, epochs=200):
    """Perceptron."""
    Y = data[:, 1]
    X = data[:, 0]
    W = np.array([1, 1])

    weights = []

    for t in range(epochs):
        print("\n\nEpoch: {}".format(t))
        grad = np.zeros(2)
        for i in range(len(X)):
            delta_f = np.array([
                                -2*X[i] * (Y[i] - W[0]*X[i] - W[1]),
                                -2 * (Y[i] - W[0]*X[i] - W[1]),
                               ])
            grad = grad + delta_f
        grad = grad / len(X)

        print("W before = {}".format(W))
        W = W - (lr * grad)
        print("W after = {}".format(W))
        weights.append(W)
        print("New Weight: {}".format(W))
        # grad = grad
        if np.linalg.norm(lr*grad) < 0.0001:
            break

    return W, np.array(weights)


def MSE_Newton(data, lr=0.01, epochs=200):
    """Perceptron."""
    Y = data[:, 1]
    X = data[:, 0]
    W = np.array([0, 0])

    weights = []

    for t in range(epochs):
        print("\n\nEpoch: {}".format(t))
        grad = np.zeros(2)
        for i in range(len(X)):
            delta_f = np.array([
                                -2*X[i] * (Y[i] - W[0]*X[i] - W[1]),
                                -2 * (Y[i] - W[0]*X[i] - W[1]),
                               ])
            H = np.array([[2*(X[i]**2), 2*X[i]],
                          [2*X[i],      2]])
            try:
                H_inv = np.linalg.inv(H)
                grad = grad + np.dot(H_inv, delta_f)
            except np.linalg.LinAlgError:
                continue

        grad = grad / len(X)

        print("W before = {}".format(W))
        W = W - (lr * grad)
        print("W after = {}".format(W))
        weights.append(W)
        print("New Weight: {}".format(W))
        # grad = grad
        # if np.linalg.norm(lr*grad) < 0.0001 and np.linalg.norm(lr*grad) > 0:
        #     break

    return W, np.array(weights)


if __name__ == "__main__":
    epochs = 200
    X = np.array([
        [2, 2],
        [3, 4],
        [6, 6],
        [1, 2],
        [4, 5],
        [4, 3],
        [5, 6],
        [2, 3],
        [6, 5],
        [7, 6],
    ])

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # W, weights = MSE_SGD(X, epochs=200)
    W, weights = MSE_Newton(X, epochs=200)

    with open("out.txt", "w") as f:
        for t in range(epochs):
            f.write("\nEpoch {}:\n".format(t))
            f.write("W = {}\n".format(weights[t]))
