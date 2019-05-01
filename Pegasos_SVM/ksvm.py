import numpy as np
import matplotlib.pyplot as plt
import pdb
from sklearn.datasets import make_moons


class MercerSVM:
    """Kernel SVM with Pegasos optimisation."""
    def __init__(self, iterations=1000, C=1, verbose=False, sigma=0.5):
        self.iter = iterations
        self.C = C
        self.sigma = sigma
        self.verbose = verbose

    def kernel(self, x):
        power = -1 * ((self.X - x)**2).sum(axis=1)
        power = power / (2 * self.sigma**2)
        out = np.exp(power)
        return out

    def fit(self, X, Y, bias=True):
        """Training."""
        # Adding bias term
        if bias:
            aug = np.ones((X.shape[0], 1))
            X = np.hstack((X, aug))
            self.bias = True

        # Shuffling the data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]

        # Finding the classes and making labels as -1, +1
        y_classes = np.unique(Y)
        if len(y_classes) != 2:
            raise ValueError("SVM is a binary classifier and accept only two classes.")

        self.class_dict = {
                    y_classes[0]: -1,
                    y_classes[1]: +1
        }
        Y_new = np.array([self.class_dict[y] for y in Y])
        Y_new = Y_new[indices]

        # Initialize W
        alpha = np.zeros(X.shape[0], dtype=float)

        # Save X and Y for kernel computations
        self.X = X
        self.Y = Y_new

        # Gradient descent
        for t in range(1, self.iter+1):
            if self.verbose:
                if t % 100 == 0:
                    print("Training iteration: ", t)
            i = np.random.choice(indices, 1)[0]
            x, y = X[i], Y_new[i]
            eta = 1 / float(self.C * t)

            kernel_val = self.kernel(x)
            # pdb.set_trace()
            score = (kernel_val * alpha * Y_new).sum()

            if y * eta * score < 1:
                alpha[i] += 1

        eta = 1 / float(self.C * t)

        # Save trained weight vector
        self.alpha = alpha * eta

    def predict(self, X):
        """Testing."""
        # Adding bias term
        if self.bias:
            aug = np.ones((X.shape[0], 1))
            X = np.hstack((X, aug))

        # Check dimensions
        assert X.shape[1] == self.X.shape[1]

        # Classification
        predictions = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            kernel_val = self.kernel(x)
            score = (kernel_val * self.alpha * self.Y).sum()
            predictions[i] = -1 if score <= 0 else 1

        # Replacing with original labels
        invert_class = dict([(self.class_dict[x], x) for x in self.class_dict])
        predictions = np.array([invert_class[x] for x in predictions])

        return predictions


def decision_boundary(model):
    """Plot the points for decision boundary."""
    X = np.linspace(model.X[:, 0].min(), model.X[:, 0].max(), 50).astype(np.float)
    Y = np.linspace(model.X[:, 1].min(), model.X[:, 1].max(), 50).astype(np.float)
    for i in X:
        for j in Y:
            x = np.array([i, j, 1])
            kernel_val = model.kernel(x)
            score = (kernel_val * model.alpha * model.Y).sum()
            if score > 0:
                plt.plot(i, j, 'b.', marker='+')
            else:
                plt.plot(i, j, 'r.', marker=1)


if __name__ == "__main__":
    X, Y = make_moons(n_samples=1000, noise=.05)
    model = MercerSVM(C=0.1, iterations=10000, verbose=True)
    model.fit(X, Y)
    ypred = model.predict(X)
    print((ypred == Y).sum())
    plt.scatter(X[:, 0], X[:, 1], color=['red' if y else 'blue' for y in Y])
    decision_boundary(model)
    plt.legend()
    plt.show()
