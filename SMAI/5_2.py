import matplotlib.pyplot as plt
import pdb


def sgd(init, n):
    x = init
    errors = []
    x_vals = []
    c = 0
    while True:
        c += 1
        df = 2 * x
        x = x - n*df
        fx = x*x
        print("X = {}, f(X) = {}".format(x, fx))
        x_vals.append(x)
        errors.append(fx)
        if fx < 0.00001:
            return errors, x_vals
        if c > 1000:
            return errors, x_vals


if __name__ == "__main__":
    print("Convergence")
    errors, X_vals = sgd(-2, 0.1)
    plt.plot(X_vals, label="Convergence", color='red')
    plt.xlabel("Iterations")
    plt.ylabel("X")
    plt.legend()
    plt.show()

    print("Divergence")
    errors, X_vals = sgd(-2, -0.1)
    plt.plot(X_vals, label="Divergence", color='blue')
    plt.xlabel("Iterations")
    plt.ylabel("X")
    plt.legend()
    plt.show()

    print("Oscillation")
    errors, X_vals = sgd(-2, 1)
    plt.plot(X_vals[0: 100], label="Oscillation", color='green')
    plt.xlabel("Iterations")
    plt.ylabel("X")
    plt.legend()
    plt.show()
