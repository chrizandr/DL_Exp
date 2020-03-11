from mnist import MNIST
import numpy as np
import pdb


def train_test_5nn(X, Y, X_test, Y_test):
    """Train and test 5 Nearest Neighbour."""
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    model.fit(X, Y)
    Y_pred = model.predict(X_test)
    acc = (Y_pred == Y_test).sum() / len(Y_test)
    return acc


def train_test_perceptron(X, Y, X_test, Y_test):
    """Train and test Perceptron."""
    from sklearn.linear_model import Perceptron
    model = Perceptron(max_iter=100)
    model.fit(X, Y)
    Y_pred = model.predict(X_test)
    acc = (Y_pred == Y_test).sum() / len(Y_test)
    return acc


def train_test_logisticereg(X, Y, X_test, Y_test):
    """Train and test Logistic Regression."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X_test)
    acc = (Y_pred == Y_test).sum() / len(Y_test)
    return acc


def train_test_linSVM(X, Y, X_test, Y_test):
    """Train and test Linear SVM."""
    from sklearn.svm import SVC
    print("Training Linear SVM")
    model = SVC(kernel='linear', max_iter=100)
    model.fit(X, Y)
    pdb.set_trace()
    Y_pred = model.predict(X_test)
    acc = (Y_pred == Y_test).sum() / len(Y_test)
    return acc


def train_test_rbfSVM(X, Y, X_test, Y_test):
    """Train and test RBF SVM."""
    from sklearn.svm import SVC
    print("Training RBF SVM")
    model = SVC(kernel='rbf', gamma='auto', max_iter=100)
    model.fit(X, Y)
    Y_pred = model.predict(X_test)
    acc = (Y_pred == Y_test).sum() / len(Y_test)
    return acc


if __name__ == "__main__":
    data_dir = "/home/chris/data/"
    mndata = MNIST(data_dir)
    X, Y = mndata.load_training()
    X, Y = np.array(X), np.array(Y)

    shuffle_indices = np.random.permutation(X.shape[0])

    X = X[shuffle_indices]
    Y = Y[shuffle_indices]

    split = int(X.shape[0] * 0.6)

    X_train, Y_train = X[0:split], Y[0:split]
    X_test, Y_test = X[split::], Y[split::]
    print("training")
    # print(train_test_5nn(X_train, Y_train, X_test, Y_test))
    # print(train_test_perceptron(X_train, Y_train, X_test, Y_test))
    # print(train_test_logisticereg(X_train, Y_train, X_test, Y_test))
    print(train_test_linSVM(X_train, Y_train, X_test, Y_test))
    print(train_test_rbfSVM(X_train, Y_train, X_test, Y_test))
    pdb.set_trace()
