import numpy as np
import pdb


def gini(Pa, Pb):
    return 1 - Pa**2 - Pb**2


def entropy(Pa, Pb):
    if Pa != 0:
        Pa = Pa * np.log(Pa)
    if Pb != 0:
        Pb = Pb * np.log(Pb)
    return -1 * (Pa + Pb)


na = 10
nb = 10

min_gini = 10000000
min_ent = 100000000
min_gini_index = (0, 0)
min_ent_index = (0, 0)

vals = []
for ka in range(1, na):
    for kb in range(1, nb):
        vals.append((ka, kb))

for i in range(10):
    choice = [vals[np.random.randint(80)] for x in range(10)]
    for ka, kb in choice:
        Pa1 = ka / float(ka + kb)
        Pb1 = kb / float(ka + kb)
        Pa2 = (na - ka) / float(na + nb - ka - kb)
        Pb2 = (nb - kb) / float(na + nb - ka - kb)
        g = gini(Pa1, Pb1) + gini(Pa2, Pb2)
        h = entropy(Pa1, Pb1) + entropy(Pa2, Pb2)

        if min_gini > g:
            min_gini = g
            min_gini_index = (ka, kb)

        if min_ent > h:
            min_ent = h
            min_ent_index = (ka, kb)

    assert min_ent_index == min_gini_index
    print("Set of values considered : ", choice)
    print("Minimum ka: {}, kb: {}".format(min_gini_index[0], min_gini_index[1]))

pdb.set_trace()
