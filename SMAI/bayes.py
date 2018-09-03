import numpy as np
import matplotlib.pyplot as plt
import operator


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
    '''Error function'''
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

    plt.scatter(X1[:, 0], X1[:, 1], color='red', label="Class 0")
    plt.scatter(X2[:, 0], X2[:, 1], color='blue', label="Class 1")
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title("Generated data")
    plt.grid()
    plt.legend()
    plt.show()

    probs = {0: 0.5, 1: 0.5}
    means, covariances = Find_Mean_Covariance(X, Y)
    predictions = Bayes_Eval(X, means, covariances, probs)

    for i in probs.keys():
        print("\nG_{}(X) is given by the following params:\n".format(i))
        print("The covariance matrix for class {} is: \n{}".format(i, covariances[i]))
        print("The mean for class {} is {}".format(i, means[i]))
        print("The prior probability for class {} is {}".format(i, probs[i]))

    print("\nClassifying\n")
    error = Bayes_Error(Y, predictions)
    print("The final error is {}%".format(error*100))
