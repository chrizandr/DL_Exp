from sklearn.neighbors import KNeighborsClassifier
from CIFAR10 import *

KNN = KNeighborsClassifier(1)
path = "/home/chris/Downloads/cifar-10-batches-py/"
Xtr, Ytr, Xte, Yte = load_Cifar(path)

KNN.fit(Xtr, Ytr)
predictions = KNN.predict(Xte)
pdb.set_trace()
