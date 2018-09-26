"""Logistic Regression two class compared with Bayes Optimal decision."""
import matplotlib.pyplot as plt
import numpy as np
import operator


def Logit(W, X):
    """Logit function."""
    return 1.0 / (1 + np.exp(-1*np.dot(X, W)))


def Logistic_Error(X, W, Y, bias=True):
    """Find error of LR."""
    n, d = X.shape
    X_aug = X
    if bias:
        X_aug = np.hstack((X, np.ones((n, 1))))

    error = 0.0
    for i, x in enumerate(X_aug):
        y_pred = 1 if Logit(W, x) > 0.5 else 0
        error += 0 if y_pred == Y[i] else 1

    return error/n


def Logistic_Regression(X, Y, lr=0.1):
    """Logistic Regression."""
    n, d = X.shape
    X_aug = np.hstack((X, np.ones((n, 1))))
    W = np.array([1, 1, 1])
    update = np.ones(W.shape)
    weights = []
    t = 0

    while np.linalg.norm(update) > 0.0001:
        update = np.zeros(W.shape)
        for i, x in enumerate(X_aug):
            # pdb.set_trace()
            y_pred = Logit(W, x)
            update += lr * (Y[i] - y_pred) * x

        update = update/X_aug.shape[0]
        W = W + update
        weights.append(W)

        if t % 1000 == 0:
            print("Epoch: {}, Cost: {}".format(t, Cost(X_aug, Y, W)))
        t += 1

    return W, np.array(weights)


def Logistic_Regression_nobias(X, Y, lr=0.1):
    """LR no bias."""
    n, d = X.shape
    X_aug = X
    W = np.array([1, 1])
    update = np.ones(W.shape)
    weights = []
    t = 0

    while np.linalg.norm(update) > 0.0001:
        update = np.zeros(W.shape)
        for i, x in enumerate(X_aug):
            # pdb.set_trace()
            y_pred = Logit(W, x)
            update += lr * (Y[i] - y_pred) * x

        update = update/X_aug.shape[0]
        W = W + update
        weights.append(W)

        if t % 1000 == 0:
            print("Epoch: {}, Cost: {}".format(t, Cost(X_aug, Y, W)))
        t += 1

    return W, np.array(weights)


def Cost(X, Y, W):
    """Cost function."""
    num_samples = X.shape[0]
    predictions = Logit(W, X)

    class1_cost = -Y*np.log(predictions)
    class2_cost = (1-Y)*np.log(1-predictions)

    cost = class1_cost - class2_cost
    cost = cost.sum()/num_samples

    return cost


def argmax(prediction):
    """Argmax."""
    return max(prediction.items(), key=operator.itemgetter(1))[0]


def Bayes_Decision(means, covariances, probs):
    """Cost function."""
    weights = {}

    for i in covariances.keys():
        inverse_covariance = np.linalg.inv(covariances[i])
        term = np.dot(means[i], inverse_covariance)
        term = np.dot(term, means[i])
        term *= -0.5
        term += np.log(probs[i])
        bias = term

        W = np.dot(inverse_covariance, means[i].T)
        weights[i] = np.concatenate((W, [bias]))

    return weights


def Find_Mean_Covariance(X, Y):
    """Mean and covariance."""
    labels = np.unique(Y)
    means = {}
    covariances = {}
    for l in labels:
        indices = (Y == l).nonzero()[0]
        X_samples = X[indices]
        means[l] = np.mean(X_samples, axis=0)
        covariances[l] = np.cov(X.T)

    return means, covariances


def decision_boundary(W):
    """Plot the points for decision boundary."""
    X = np.linspace(-10, 10, 30).astype(np.float)
    Y = -1*(W[0]/W[1])*X - (W[2]/W[1])
    return X, Y


if __name__ == "__main__":
    X1 = np.random.normal(loc=0, scale=1, size=(500, 2))
    Y1 = np.zeros(500, dtype=np.int)
    X2 = np.random.normal(loc=2, scale=2, size=(500, 2))
    Y2 = np.ones(500, dtype=np.int)
    probs = {0: 0.5, 1: 0.5}

    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    means, covariances = Find_Mean_Covariance(X, Y)
    W_bayes = Bayes_Decision(means, covariances, probs)

    W_bias, _ = Logistic_Regression(X, Y)
    W_nobias, _ = Logistic_Regression_nobias(X, Y)

    error_bias = Logistic_Error(X, W_bias, Y, bias=True)
    error_nobias = Logistic_Error(X, W_nobias, Y, bias=False)
    print("Error with bias: ", error_bias)
    print("Error without bias: ", error_nobias)
    # Plot decision boundary
    W_nobias = np.concatenate((W_nobias, [0]))

    X, y = decision_boundary(W_bayes[0] - W_bayes[1])
    plt.plot(X, y, color='pink', label="Bayes classifier")
    X, y = decision_boundary(W_bias)
    plt.plot(X, y, color='green', label="Logistic Regression")
    X, y = decision_boundary(W_nobias)
    plt.plot(X, y, color='yellow', label="Logistic Regression no bias")
    plt.scatter(X1[:, 0], X1[:, 1], color='red', label="Class 0")
    plt.scatter(X2[:, 0], X2[:, 1], color='blue', label="Class 1")
    plt.grid()
    plt.legend()
    plt.show()

    X1 = np.random.normal(loc=0, scale=1, size=(500, 2))
    Y1 = np.zeros(500, dtype=np.int)
    X2 = np.random.normal(loc=2, scale=2, size=(500, 2))
    Y2 = np.ones(500, dtype=np.int)
    probs = {0: 0.5, 1: 0.5}

    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    W_bias, _ = Logistic_Regression(X, Y)
    W_nobias, _ = Logistic_Regression_nobias(X, Y)

    print("Resampling")
    error_bias = Logistic_Error(X, W_bias, Y, bias=True)
    error_nobias = Logistic_Error(X, W_nobias, Y, bias=False)

    print("Error with bias: ", error_bias)
    print("Error without bias: ", error_nobias)
