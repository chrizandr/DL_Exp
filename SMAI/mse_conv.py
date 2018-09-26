"""MSE Convergence."""
import matplotlib.pyplot as plt


def sgd(x, n):
    """Gradient descent."""
    errors = []
    x_vals = []
    while True:
        df = 2 * x
        x = x - n*df
        fx = x*x
        print("X = {}, f(X) = {}".format(x, fx))
        x_vals.append(x)
        errors.append(fx)
        if fx < 0.00001:
            return errors, x_vals


if __name__ == "__main__":
    errors, X_vals = sgd(-2, 0.1)

    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title("Convergence at Error = 0.00001")
    plt.show()

    plt.plot(X_vals)
    plt.xlabel('Iterations')
    plt.ylabel('X_value')
    plt.title("Converges at x approximately equal to 0")
    plt.show()
