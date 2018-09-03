import numpy as np
import matplotlib.pyplot as plt
import operator

import pdb


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
    """Perceptron."""
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
    '''Cost function'''
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


def Bayes_Error(Y, predictions):
    """Bayes."""
    samples = Y.shape[0]
    error = 0.0
    for i, y in enumerate(Y):
        y_pred = argmax(predictions[i])
        error += 0 if y_pred == y else 1

    return error/samples


def Bayes_Eval(X, means, covariances, probs):
    '''Cost function'''
    addition_term = {}
    inverse_covariance = {}
    predictions = []

    for i in covariances.keys():
        term = -0.5 * np.log(np.linalg.norm(covariances[i]))
        term += np.log(probs[i])
        addition_term[i] = term
        inverse_covariance[i] = np.linalg.inv(covariances[i])

    for x in X:
        prediction = {}
        for i in means.keys():
            score = np.dot((x - means[i]).T, inverse_covariance[i])
            score = np.dot(score, x - means[i])
            score = -0.5*score
            score += addition_term[i]
            prediction[i] = score
        predictions.append(prediction)

    return predictions


def Find_Mean_Covariance(X, Y):
    labels = np.unique(Y)
    means = {}
    covariances = {}
    for l in labels:
        indices = (Y == l).nonzero()[0]
        X_samples = X[indices]
        means[l] = np.mean(X_samples, axis=0)
        covariances[l] = np.cov(X.T)

    return means, covariances


if __name__ == "__main__":
    X1 = np.random.normal(loc=0, scale=1, size=(500, 2))
    Y1 = np.zeros(500, dtype=np.int)
    X2 = np.random.normal(loc=2, scale=2, size=(500, 2))
    Y2 = np.ones(500, dtype=np.int)

    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    W_bias, _ = Logistic_Regression(X, Y)
    W_nobias, _ = Logistic_Regression_nobias(X, Y)

    error_bias = Logistic_Error(X, W_bias, Y, bias=True)
    error_nobias = Logistic_Error(X, W_nobias, Y, bias=False)

    pdb.set_trace()
