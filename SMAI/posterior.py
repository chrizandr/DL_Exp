import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def PDF(X, mean, covariance):
    p = 1 / np.sqrt(2 * np.pi)
    p = p * np.exp(-1*(X-mean)**2 / (2*covariance**2))
    return p


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

    score1 = np.zeros((200, 200), dtype=np.float)
    score2 = np.zeros((200, 200), dtype=np.float)
    addition_term = {}
    inverse_covariance = {}

    k1, k2 = means.keys()

    for i in covariances.keys():
        term = -0.5 * np.log(np.linalg.norm(covariances[i]))
        term += np.log(probs[i])
        addition_term[i] = term
        inverse_covariance[i] = np.linalg.inv(covariances[i])

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            p = np.array([x, y])

            score_1 = np.dot((p - means[k1]).T, inverse_covariance[k1])
            score_1 = np.dot(score_1, p - means[k1])
            score_1 = -0.5*score_1
            score_1 += addition_term[k1]

            score_2 = np.dot((p - means[k2]).T, inverse_covariance[k2])
            score_2 = np.dot(score_2, p - means[k2])
            score_2 = -0.5*score_2
            score_2 += addition_term[k2]

            score1[j, i] = score_1
            score2[j, i] = score_2

    return x_mesh, y_mesh, score1, score2


if __name__ == "__main__":
    X = np.linspace(-10, 20, 100)

    P_X_W1 = PDF(X, 1, 2)
    P_X_W2 = PDF(X, 3, 1)

    # Assume equal prior probabilities
    P_W1 = 0.6
    P_W2 = 0.4

    P_X = P_X_W1 * P_W1 + P_X_W2 * P_W2

    P_W1_X = P_X_W1 * P_W1 / P_X
    P_W2_X = P_X_W2 * P_W2 / P_X

    # plt.plot(X, P_X_W1, color='red', label="P(X|W1)")
    # plt.plot(X, P_X_W2, color='blue', label="P(X|W2)")
    # plt.plot(X, P_X_W1 * P_W1, color='blue', label="P(X|W1) x P(W1)")
    # plt.plot(X, P_X_W2 * P_W2, color='blue', label="P(X|W2) x P(W2)")
    # plt.plot(X, P_X, color='green', label="P(X)")
    plt.plot(X, P_W1_X, color='red', label="P(W1|X)")
    plt.plot(X, P_W2_X, color='blue', label="P(W2|X)")
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
    pdb.set_trace()

    means, covariances = Find_Mean_Covariance(X, Y)
    x_vals, y_vals, score1, score2 = decision_boundary(means, covariances, probs)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_vals, y_vals, score1, linewidth=0, antialiased=False)
    ax.set_zlabel('Decision Boundary')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Decision Boundary")
    plt.show()

    pdb.set_trace()
