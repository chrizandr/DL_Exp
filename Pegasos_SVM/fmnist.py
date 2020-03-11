from reader import load_mnist
import numpy as np
from svm import SVM
from ksvm import MercerSVM
import pdb


def load_data(folder="data/"):
    """Randomly choose two labels from FashionMNIST and return."""
    X_train, y_train = load_mnist(folder, kind='train')
    X_test, y_test = load_mnist(folder, kind='t10k')
    # Randomly select two class labels
    y_classes = np.unique(y_train)
    y_select = np.random.choice(y_classes, 2)

    indices = []
    for i, y in enumerate(y_train):
        if y in y_select:
            indices.append(i)
    indices = np.array(indices)
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]

    indices = []
    for i, y in enumerate(y_test):
        if y in y_select:
            indices.append(i)
    indices = np.array(indices)
    np.random.shuffle(indices)
    X_test, y_test = X_test[indices], y_test[indices]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    # Linear SVM
    model = SVM(C=0.005, iterations=1000, verbose=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).sum() / float(len(y_test)) * 100
    print("Accuracy with Linear SVM: {}%".format(acc))

    # Kernel SVM
    model = MercerSVM(C=0.005, iterations=100, sigma=0.5, verbose=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).sum() / float(len(y_test)) * 100
    print("Accuracy with Kernel SVM: {}%".format(acc))
