import numpy as np


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
    epochs = 200
    X = np.array([
        [2, 4],
        [3, 5],
        [6, 8],
        [1, 2],
        [4, 7],
        [4, 2],
        [5, 1],
        [8, 3],
        [6, 4],
        [7, 5],
    ])
    plt.scatter()
    Y = np.array([-1, -1, 1, -1, 1, 1])

    W, weights = Perceptron(X, Y, epochs=200)

    with open("out.txt", "w") as f:
        for t in range(epochs):
            f.write("\nEpoch {}:\n".format(t))
            for i in range(len(X)):
                f.write("W = {}\n".format(weights[t+i]))
