"""SGD vs Newton Descent."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def Activation(W, x):
    """Activation function."""
    return np.dot(W, x)


def MSE_SGD(data, lr=0.01, epochs=200):
    """Perceptron."""
    Y = data[:, 1]
    X = data[:, 0]
    W = np.array([1, 1])

    weights = []
    grad = np.array([10, 10])
    t = 0

    while np.linalg.norm(lr*grad) > 0.0001:
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

        t += 1

    return W, np.array(weights), t


def MSE_Newton(data, epochs=200):
    """Perceptron."""
    Y = data[:, 1]
    X = data[:, 0]
    W = np.array([1, 1])

    weights = []
    grad = np.array([10, 10])
    t = 0

    while np.linalg.norm(grad) > 0.0001:
        print("\n\nEpoch: {}".format(t))
        H = np.zeros((2, 2))
        delta_f = np.zeros(2)
        for i in range(len(X)):
            delta_f += np.array([
                                -2*X[i] * (Y[i] - W[0]*X[i] - W[1]),
                                -2 * (Y[i] - W[0]*X[i] - W[1]),
                                ])
            H += np.array([[2*(X[i]**2), 2*X[i]],
                           [2*X[i],      2]])
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue

        grad = np.dot(H_inv, delta_f)
        grad = grad / len(X)

        print("W before = {}".format(W))
        W = W - grad
        print("W after = {}".format(W))
        weights.append(W)
        print("New Weight: {}".format(W))

        t += 1

    return W, np.array(weights), t


def error(W, X, Y):
    """Find error."""
    error = 0
    for i in range(len(X)):
        error += (Y[i] - W[0]*X[i] - W[1])**2
    return error


if __name__ == "__main__":
    # Data
    X = np.array([
        [2, 2],
        [3, 4],
        [1, 2],
        [4, 5],
        [6, 6],
        [4, 3],
        [5, 6],
        [2, 3],
        [6, 5],
        [7, 6],
    ])

    plt.scatter(X[:, 0], X[:, 1], color='red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    # Training
    W_sgd, weights_sgd, t_sgd = MSE_SGD(X, epochs=200)
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
