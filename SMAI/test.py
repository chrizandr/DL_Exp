import numpy as np


def predict(x, W):
    """Activation function."""
    return 1.0 / (1 + np.exp(-1*np.dot(W, x)))


def cost_function(features, labels, weights):
    '''
    Using Mean Absolute Error

    Features:(100,3)
    Labels: (100,1)
    Weights:(3,1)
    Returns 1D matrix of predictions
    Cost = ( log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
    '''
    observations = len(labels)
    predictions = predict(features, weights)
    class1_cost = -labels*np.log(predictions)
    class2_cost = (1-labels)*np.log(1-predictions)
    cost = class1_cost - class2_cost
    cost = cost.sum()/observations

    return cost


def update_weights(features, labels, weights, lr):
    '''
    Vectorized Gradient Descent

    Features:(200, 3)
    Labels: (200, 1)
    Weights:(3, 1)
    '''
    N = len(features)
    predictions = predict(features, weights)
    gradient = np.dot(features.T,  predictions - labels)
    gradient /= N
    gradient *= lr
    weights -= gradient

    return weights


def train(features, labels, weights, lr, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        if i % 1000 == 0:
            print "iter: "+str(i) + " cost: "+str(cost)

    return weights, cost_history


def read_data(filename):
    f = open(filename)
    features = []
    labels = []
    for line in f:
        content = line.strip.split(',')
        features.append([float(x) for x in content[0:-1]])
        labels.append(int(content[-1]))

    f.close()
    return np.array(features), np.array(labels)

folder = "/home/chris/Downloads/data.csv"
data = read_data(folder)


X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [7, 6],
    [6, 5],
    [4, 3],
    [3, 2],
    [2, 1],
    ])

Y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
W = np.array([1, 1, 1])
