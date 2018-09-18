import matplotlib.pyplot as plt
import numpy as np


def evaluate(x):
    if x < 0:
        return np.sin(np.pi * x)
    else:
        return x**2 - 2


def sgd(init, n):
    x = init
    errors = []
    x_vals = []
    c = 0
    while True:
        c += 1
        if x < 0:
            df = np.pi * np.cos(np.pi * x)
        else:
            df = 2 * x

        x = x - n*df
        fx = evaluate(x)
        print("X = {}, f(X) = {}".format(x, fx))
        x_vals.append(x)
        errors.append(fx)
        if fx < -1.9:
            return errors, x_vals
        if c > 1000:
            return errors, x_vals


if __name__ == "__main__":
    X = 2
    errors, X_vals = sgd(X, 0.1)
    plt.plot(X_vals, color='red')
    plt.xlabel("Iterations")
    plt.ylabel("X")
    plt.title("Convergence at X={}".format(X))
    plt.legend()
    plt.show()

    X = -1
    errors, X_vals = sgd(X, 0.1)
    plt.plot(X_vals, color='red')
    plt.xlabel("Iterations")
    plt.ylabel("X")
    plt.title("No Convergence at X={}".format(X))
    plt.legend()
    plt.show()

    X = -4
    errors, X_vals = sgd(X, 0.1)
    plt.plot(X_vals, color='red')
    plt.xlabel("Iterations")
    plt.ylabel("X")
    plt.title("No Convergence at X={}".format(X))
    plt.legend()
    plt.show()
