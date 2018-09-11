import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Find_Mean_Covariance(X, Y):
    labels = np.unique(Y)
    means = {}
    covariances = {}
    for l in labels:
        indices = (Y == l).nonzero()[0]
        X_samples = X[indices]
        means[l] = np.mean(X_samples, axis=0)
        covariances[l] = np.cov(X_samples.T)

    return means, covariances


def decision_boundary(means, covariances, probs):
    assert len(means.keys()) == 2
    X = np.linspace(-100, 100, 200)
    Y = np.linspace(-100, 100, 200)
    x_mesh, y_mesh = np.meshgrid(X, Y)

    dbound = np.zeros((200, 200), dtype=np.float)
    addition_term = {}
    inverse_covariance = {}

    k1, k2 = means.keys()

    for i in covariances.keys():
        term = -0.5 * np.log(np.linalg.norm(covariances[i]))
        term += np.log(probs[i])
        addition_term[i] = term
        inverse_covariance[i] = np.linalg.inv(covariances[i])

    w2_k1 = np.dot(inverse_covariance[k1], means[k1])
    w0_k2 = (-0.5)*np.dot(np.dot(np.transpose(means[k1]), inverse_covariance[k1]), means[k1]) + np.log(probs[k1])
    w1_k1 = -0.5*inverse_covariance[k1]
    w2_k1 = np.dot(inverse_covariance[k1], means[k1])
    w0_k1 = (-0.5)*np.dot(np.dot(np.transpose(means[k1]), inverse_covariance[k1]), means[k1]) + np.log(probs[k1]) - 0.5*np.log(np.linalg.norm(covariances[k1]))

    w2_k2 = np.dot(inverse_covariance[k2], means[k2])
    w0_k2 = (-0.5)*np.dot(np.dot(np.transpose(means[k2]), inverse_covariance[k2]), means[k2]) + np.log(probs[k2])
    w1_k2 = -0.5*inverse_covariance[k2]
    w2_k2 = np.dot(inverse_covariance[k2], means[k2])
    w0_k2 = (-0.5)*np.dot(np.dot(np.transpose(means[k2]), inverse_covariance[k2]), means[k2]) + np.log(probs[k2]) - 0.5*np.log(np.linalg.norm(covariances[k2]))

    w1 = w1_k1 - w1_k2
    w2 = w2_k1 - w2_k2
    w0 = w0_k1 - w0_k2

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            p = np.array([x, y])
            dbound[j, i] = np.dot(np.dot(p, w1), p) + np.dot(np.transpose(w2), np.transpose(p)) + w0

    return x_mesh, y_mesh, dbound


if __name__ == "__main__":
    X1 = np.random.normal(loc=2, scale=1, size=(100, 2))
    Y1 = np.zeros(100, dtype=np.int)
    X2 = np.random.normal(loc=2, scale=5, size=(100, 2))
    Y2 = np.ones(100, dtype=np.int)

    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    # Assume equal prior probabilities
    probs = {0: 0.5, 1: 0.5}

    means, covariances = Find_Mean_Covariance(X, Y)

    x_vals, y_vals, dbound = decision_boundary(means, covariances, probs)

    plt.scatter(X1[:, 0], X1[:, 1], color='blue', label='Class A')
    plt.scatter(X2[:, 0], X2[:, 1], color='red', label='Class B')
    plt.contour(x_vals, y_vals, dbound, levels=[0], label='Decision Boundary', color='green')
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision Boundary')
    plt.show()

    pdb.set_trace()
