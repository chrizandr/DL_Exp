import numpy as np
from collections import Counter
from CIFAR10 import *
import pdb
'''Calculates the Euclidean distances between the training data the testing data'''
def Euclid(Xtr , X):
    # Using numpy functions for optimal implementations
    return np.sum(np.abs(Xtr - X), axis = 1)

'''Calculates the Manhattan distances between the training data the testing data'''
def Manhattan(Xtr, X):
    # Using numpy functions for optimal implementations
    return np.sqrt(np.sum(np.square(Xtr - X), axis = 1))

'''Calculates the accuracy of the predictions'''
def Accuracy(Ytr , Y):
    return np.mean(Ytr == Y)

class Knn(object):

    '''
    Constructor for the KNN classifier:
    Parameters : k - Number of Neighbors to consider
    dist - function to calculate distances between an NxD matrix and 1XD matrix, row wise
    '''
    def __init__(self, k, dist):
        self.k = k
        self.metric = dist

    '''
    Parameters : X - Training data ; Y - Training classes;
    '''
    def train(self, X, Y):
        # Store the data in Xtr and Ytr for classification in future
        self.Xtr = X
        self.Ytr = Y

    '''
    Parameters : X - Testing data
    '''
    def predict(self, X):
        # Create a vector for predictions
        prediction = np.zeros((X.shape[0],1),dtype = self.Ytr.dtype)
        # Predict each row one by one and store it in prediction
        for i in range(X.shape[0]):
            ranks = list()
            # Evaluating the distance between current test case and training data set
            distances = self.metric(self.Xtr , X[i])
            # Assigning labels to each of the training examples along with the distance from current case
            labels = [(distances[j] , self.Ytr[j]) for j in range(self.Xtr.shape[0])]
            # Sorting the distances and taking only the first k-Neighbors
            labels = sorted(labels , key = lambda tup: tup[0])[0:self.k]
            # Converting labels back into a list of classes
            labels = [x[0] for x in labels]
            # Using the most frequent label as the prediction
            prediction[i] = Counter(labels).most_common(1)[0][0]

        return prediction


KNeighbors = Knn(1, Manhattan)
path = "/home/chris/Downloads/cifar-10-batches-py/"
Xtr, Ytr, Xte, Yte = load_Cifar(path)
KNeighbors.train(Xtr , Ytr)
prediction = KNeighbors.predict(Xte)

pdb.set_trace()
