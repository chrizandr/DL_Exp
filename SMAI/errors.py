import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Activation_Log(W, X):
    """Activation function."""
    return 1.0 / (1 + np.exp(-1*np.dot(X, W)))


def Activation_Linear(W, X):
    """Activation function."""
    return np.dot(X, W)


def Error_Linear(X, Y, W=np.array([3, -2])):
    """Error."""
    n, d = X.shape
    X_aug = np.hstack((X, np.ones((n, 1))))
    # Initialised with line x1 = 1
    errors = []

    for i, x in enumerate(X_aug):
        y_pred = Activation_Linear(W, x)
        error = (Y[i] - y_pred)**2
        errors.append(error)

    return errors


def Error_Log(X, Y, W=np.array([3, -2])):
    """Error."""
    n, d = X.shape
    X_aug = np.hstack((X, np.ones((n, 1))))
    # Initialised with line x1 = 1
    errors = []

    for i, x in enumerate(X_aug):
        y_pred = Activation_Log(W, x)
        error = (Y[i] - y_pred)**2
        errors.append(error)

    return errors


if __name__ == "__main__":
    epochs = 200
    X = np.linspace(-10, 10, num=1000).reshape(1000, 1)
    Y_1 = np.zeros(1000)
    Y_2 = np.ones(1000)
    Y = np.random.choice([0, 1], size=(1000,))

    lin_loss_1 = Error_Linear(X, Y_1)
    lin_loss_2 = Error_Linear(X, Y_2)

    log_loss_1 = Error_Log(X, Y_1)
    log_loss_2 = Error_Log(X, Y_2)

    # fig = plt.figure()
    plt.plot(X.reshape(1000), lin_loss_1, color='red', label="Class -1")
    plt.plot(X.reshape(1000), lin_loss_2, color='blue', label="Class +1")
    plt.xlabel("X")
    plt.ylabel("Error")
    plt.title("Linear Loss")
    plt.legend()
    plt.show()

    plt.plot(X.reshape(1000), log_loss_1, color='red', label="Class -1")
    plt.plot(X.reshape(1000), log_loss_2, color='blue', label="Class +1")
    plt.xlabel("X")
    plt.ylabel("Error")
    plt.title("Log Loss")
    plt.legend()
    plt.show()

    w1 = np.linspace(-10, 10, 30)
    w2 = np.linspace(-10, 10, 30)
    w1_mesh, w2_mesh = np.meshgrid(w1, w2)

    errors = np.array([[sum(Error_Linear(X, Y, W=np.array([w_1, w_2]))) for w_2 in w2] for w_1 in w1])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(w1_mesh, w2_mesh, errors, linewidth=0, antialiased=False)
    ax.set_zlabel('Errors')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title("Error Surface Linear")
    plt.show()

    errors = np.array([[sum(Error_Log(X, Y, W=np.array([w_1, w_2]))) for w_2 in w2] for w_1 in w1])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(w1_mesh, w2_mesh, errors, linewidth=0, antialiased=False)
    ax.set_zlabel('Errors')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title("Error Surface Log")
    plt.show()
