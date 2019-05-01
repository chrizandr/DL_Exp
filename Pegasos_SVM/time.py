import time
import numpy as np
from svm import SVM
from ksvm import MercerSVM
import matplotlib.pyplot as plt
import pdb


if __name__ == "__main__":
    # time_taken = []
    # # Dimension
    # for i in range(2, 1000, 10):
    #     print("Model dimensions ", i)
    #     X1 = np.random.normal(loc=0, scale=1, size=(500, i))
    #     Y1 = np.zeros(500, dtype=np.int)
    #     X2 = np.random.normal(loc=5, scale=1, size=(500, i))
    #     Y2 = np.ones(500, dtype=np.int)
    #     X = np.vstack((X1, X2))
    #     Y = np.append(Y1, Y2)
    #     model = SVM(iterations=10000)
    #     start = time.time()
    #     model.fit(X, Y)
    #     end = time.time()
    #     time_taken.append(end - start)
    # plt.plot(range(2, 1000, 10), time_taken, label="Dimension vs Time")
    # plt.xlabel("Dimensions")
    # plt.ylabel("Time")
    # plt.show()
    # pdb.set_trace()

    # time_taken = []
    # # Dimension, kernel SVM
    # for i in range(2, 100):
    #     print("Model dimensions ", i)
    #     X1 = np.random.normal(loc=0, scale=1, size=(500, i))
    #     Y1 = np.zeros(500, dtype=np.int)
    #     X2 = np.random.normal(loc=5, scale=1, size=(500, i))
    #     Y2 = np.ones(500, dtype=np.int)
    #     X = np.vstack((X1, X2))
    #     Y = np.append(Y1, Y2)
    #     model = MercerSVM(iterations=100)
    #     start = time.time()
    #     model.fit(X, Y)
    #     end = time.time()
    #     time_taken.append(end - start)
    # plt.plot(range(2, 100), time_taken, label="Dimension vs Time")
    # plt.xlabel("Dimensions")
    # plt.ylabel("Time")
    # plt.show()
    # pdb.set_trace()

    time_taken = []
    # Samples
    for i in range(2, 1000, 10):
        print("Model samples ", i)
        X1 = np.random.normal(loc=0, scale=1, size=(i//2, 2))
        Y1 = np.zeros(500, dtype=np.int)
        X2 = np.random.normal(loc=5, scale=1, size=(i//2, 2))
        Y2 = np.ones(500, dtype=np.int)
        X = np.vstack((X1, X2))
        Y = np.append(Y1, Y2)
        model = MercerSVM(iterations=10000)
        start = time.time()
        model.fit(X, Y)
        end = time.time()
        time_taken.append(end - start)
    plt.plot(range(2, 1000, 10), time_taken, label="Samples vs Time")
    plt.xlabel("Samples")
    plt.ylabel("Time")
    plt.show()
    pdb.set_trace()
