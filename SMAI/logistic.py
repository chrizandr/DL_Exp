import numpy as np
# import pdb
import matplotlib.pyplot as plt


def Activation(W, X):
    """Activation function."""
    return 1.0 / (1 + np.exp(-1*np.dot(X, W)))


def Logistic_Regression(X, Y, lr=0.1, epochs=200):
    """Perceptron."""
    n, d = X.shape
    X_aug = np.hstack((X, np.ones((n, 1))))
    # Initialised with line x1 = 1
    W = np.array([1, 1, 1])
    update = np.ones(W.shape)
    weights = []
    t = 0

    while np.linalg.norm(update) > 0.0001:
        update = np.zeros(W.shape)
        for i, x in enumerate(X_aug):
            # pdb.set_trace()
            y_pred = Activation(W, x)
            update += lr * (Y[i] - y_pred) * x

        update = update/X_aug.shape[0]
        W = W + update
        weights.append(W)

        if t % 1000 == 0:
            print("Epoch: {}, Error: {}".format(t, Error(X_aug, Y, W)))
        t += 1

    return W, np.array(weights)


def Error(X, Y, W):
    '''Error function'''
    num_samples = X.shape[0]
    predictions = Activation(W, X)

    class1_cost = -Y*np.log(predictions)
    class2_cost = (1-Y)*np.log(1-predictions)

    cost = class1_cost - class2_cost
    cost = cost.sum()/num_samples

    return cost


if __name__ == "__main__":
    epochs = 200
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [7, 6],
        [6, 5],
        [4, 3],
        [3, 2],
        [2, 1],
        ])

    Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    plt.scatter(X[:, 0], X[:, 1], color=['red' if x > 0 else 'blue' for x in Y])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    W, weights = Logistic_Regression(X, Y, epochs=200)

    with open("out.txt", "w") as f:
        for t in range(epochs):
            f.write("\nEpoch {}:\n".format(t))
            f.write("W = {}\n".format(weights[t]))
