"""Q3."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pdb


def MSE_SGD(data, lr=0.01):
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
            delta_f = -1 * Y[i] * X[i]
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
        error += (1 - Y[i] * np.dot(W.T, X[i]))
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
    W, Weights, t = MSE_SGD((X, Y))
    errors = [error(w, X, Y) for w in Weights]
    pdb.set_trace()
    W, Weights, t = MSE_SGD((X, Y_dash))
    errors = [error(w, X, Y) for w in Weights]
    W_newt, weights_newt, t_newt = MSE_Newton(X, epochs=200)

    with open("sgd.txt", "w") as f:
        for t in range(t_sgd):
            f.write("\nEpoch {}:\n".format(t))
            f.write("W = {}\n".format(weights_sgd[t]))

    with open("newt.txt", "w") as f:
        for t in range(t_newt):
            f.write("\nEpoch {}:\n".format(t))
            f.write("W = {}\n".format(weights_newt[t]))

    # Convergence
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(range(t_sgd), weights_sgd[:, 0], weights_sgd[:, 1], color='red', label="Gradient Descent")
    ax.plot(range(t_newt), weights_newt[:, 0], weights_newt[:, 1],  color='blue', label="Newton's Method")
    ax.set_xlabel('Epochs')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$\theta_2$')
    ax.legend()
    plt.show()

    errors_sgd = [error(w, X[:, 0], X[:, 1]) for w in weights_sgd]
    errors_newt = [error(w, X[:, 0], X[:, 1]) for w in weights_newt]

    plt.plot(range(t_sgd), errors_sgd,  color='red', label="Gradient Descent")
    plt.plot(range(t_newt), errors_newt,  color='blue', label="Newton's Method")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    # Error Surface
    opt_weight = weights_newt[-1]
    w1 = np.linspace(opt_weight[0] - 1, opt_weight[0] + 1, 30)
    w2 = np.linspace(opt_weight[1] - 1, opt_weight[1] + 1, 30)
    w1_mesh, w2_mesh = np.meshgrid(w1, w2)

    errors = np.array([[error([w_1, w_2], X[:, 0], X[:, 1]) for w_2 in w2] for w_1 in w1])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(w1_mesh, w2_mesh, errors, linewidth=0, antialiased=False)
    ax.set_zlabel('Errors')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title("Error Surface")
    plt.show()
