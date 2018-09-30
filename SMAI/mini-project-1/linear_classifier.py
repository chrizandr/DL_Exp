"""Linear Softmax Classification."""
from PIL import Image
import numpy as np
import sys


def read_data(filepath, type_="train", class_num=None):
    """Read different types of data from different files."""
    if type_ == "train":
        return read_train(filepath)
    elif type_ == "test":
        return read_test(filepath)
    elif type_ == "ground" and class_num is not None:
        return read_ground_truth(filepath, class_num)
    else:
        raise ValueError("Invalid input for the file type")


def read_train(filepath):
    """Read data from train file."""
    f = open(filepath)
    X = []
    Y = []
    class_num = {}
    class_counter = 0
    for line in f:
        data = line.strip().split(" ")
        img = Image.open(data[0]).convert("L")
        img.thumbnail((64, 64), Image.ANTIALIAS)
        img = np.array(img)
        X.append(img.flatten())
        if data[1] not in class_num:
            class_num[data[1]] = class_counter
            class_counter += 1
        Y.append(class_num[data[1]])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, class_num


def read_test(filepath):
    """Read sample from test file."""
    f = open(filepath)
    X = []
    for line in f:
        data = line.strip()
        img = Image.open(data).convert("L")
        img.thumbnail((64, 64), Image.ANTIALIAS)
        img = np.array(img)
        X.append(img.flatten())
    X = np.array(X)
    return X


def read_ground_truth(filepath, class_num):
    """Read ground truth values, same format as output."""
    f = open(filepath)
    Y = []
    for line in f:
        data = line.strip()
        Y.append(class_num[data])
    Y = np.array(Y)
    return Y


def PCA(X, n_components):
    """PCA reduction, reduce to n_components"""
    X = X.astype(np.float)
    X -= np.mean(X, axis=0)
    covariance = np.cov(X, rowvar=False)
    evals, evecs = np.linalg.eigh(covariance)

    # Sort Eigne vectors and values
    indices = np.argsort(evals)[::-1]
    evals = evals[indices]
    evecs = evecs[:, indices]

    max_evec = evecs[:, 0:n_components]
    if evals.dtype == np.complex128:
        max_evec = max_evec.real
    X = np.dot(X, max_evec)
    return X, max_evec


def train_classifier(X, Y, epochs=200, lr=0.001, batch_size=20):
    classes = np.unique(Y)
    N, d = X.shape

    W = 0.0001 * np.random.randn(d, len(classes))
    b = 0.0001 * np.random.randn(len(classes))

    losses = []

    for i in range(epochs):
        shuffle_indices = np.random.permutation(N)
        X_s, Y_s,  = X[shuffle_indices], Y[shuffle_indices]
        avg_loss = 0
        for i in range(N // batch_size):
            X_batch = X_s[batch_size*i: batch_size*(i+1)]
            Y_batch = Y_s[batch_size*i: batch_size*(i+1)]
            loss, dW, db = compute_loss_gradient(X_batch, Y_batch, W, b)
            avg_loss += loss
            W += -1 * dW * lr
            b += -1 * db * lr

        X_batch = X_s[batch_size*(N//batch_size):N]
        Y_batch = Y_s[batch_size*(N//batch_size):N]
        loss, dW, db = compute_loss_gradient(X_batch, Y_batch, W, b)
        avg_loss += loss
        W += -1 * dW * lr
        b += -1 * db * lr
        avg_loss = avg_loss / (N//batch_size)
        losses.append(avg_loss)

    return W, b, losses


def predict(X, W, b):
    scores = np.dot(X, W) + b
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return np.argmax(scores, axis=1)


def compute_loss_gradient(X, Y, W, b):
    N, d = X.shape
    scores = np.dot(X, W) + b

    probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(np.log(probs[list(range(N)), Y])) / N

    dscores = probs.copy()
    dscores[list(range(N)), Y] -= 1
    dscores /= N

    dW = X.T.dot(dscores)
    db = np.sum(dscores, axis=0)

    return loss, dW, db


if __name__ == "__main__":
    assert len(sys.argv) > 1

    data_path = sys.argv[1]
    test_path = sys.argv[2]

    X, Y, class_num = read_data(data_path, type_="train")
    reverse_classnum = {class_num[k]: k for k in class_num}

    X_test = read_data(test_path, type_="test")

    X_features, max_evec = PCA(X, n_components=32)
    X_features_test = np.dot(X_test, max_evec)

    W, b, losses = train_classifier(X_features, Y, epochs=2000, lr=5e-7, batch_size=20)

    predictions = predict(X_features_test, W, b)

    for p in predictions:
        print(reverse_classnum[p])
