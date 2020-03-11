import numpy as np
import matplotlib.pyplot as plt
import pdb


class SVM:
    """Primal SVM with Pegasos optimisation."""
    def __init__(self, iterations=1000, C=1, verbose=False):
        self.iter = iterations
        self.C = C
        self.verbose = verbose

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
        W = np.zeros(X.shape[1], dtype=float)

        # Gradient descent
        for t in range(1, self.iter+1):
            if self.verbose:
                if t % 100 == 0:
                    print("Training iteration: ", t)
            i = np.random.choice(indices, 1)[0]
            x, y = X[i], Y_new[i]
            eta = 1 / float(self.C * t)

            score = np.dot(W, x)

            if y * score < 1:
                W = (1 - eta * self.C) * W + (eta * y) * x
            else:
                W = (1 - eta * self.C) * W

        # Save trained weight vector
        self.W = W

    def predict(self, X):
        """Testing."""
        # Adding bias term
        if self.bias:
            aug = np.ones((X.shape[0], 1))
            X = np.hstack((X, aug))

        # Check dimensions
        assert X.shape[1] == self.W.shape[0]
        # Dot product
        score = np.dot(X, self.W)
        # Classification
        predictions = [-1 if x < 0 else 1 for x in score]
        # Replacing with original labels
        invert_class = dict([(self.class_dict[x], x) for x in self.class_dict])
        predictions = np.array([invert_class[x] for x in predictions])

        return predictions


def decision_boundary(W):
    """Plot the points for decision boundary."""
    X = np.linspace(-10, 10, 30).astype(np.float)
    Y = -1*(W[0]/W[1])*X - (W[2]/W[1])
    plt.plot(X, Y, color='pink', label="SVM boundary")


if __name__ == "__main__":
    X1 = np.random.normal(loc=0, scale=1, size=(500, 2))
    Y1 = np.zeros(500, dtype=np.int)
    X2 = np.random.normal(loc=5, scale=1, size=(500, 2))
    Y2 = np.ones(500, dtype=np.int)
    X = np.vstack((X1, X2))
    Y = np.append(Y1, Y2)

    plt.scatter(X1[:, 0], X1[:, 1], color='red', label='class A')
    plt.scatter(X2[:, 0], X2[:, 1], color='blue', label='class B')

    model = SVM(iterations=10000)
    model.fit(X, Y)
    y_pred = model.predict(X)
    print((y_pred == Y).sum())
    decision_boundary(model.W)
    plt.legend()
    plt.show()
