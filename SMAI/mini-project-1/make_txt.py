import numpy as np
import os

if __name__ == "__main__":
    data_path = "dataset/"
    files = os.listdir(data_path)

    classes = [int(x.split('.')[0].split('_')[0]) for x in files]
    classes = np.array(classes)

    train_files = []
    train_classes = []
    test_files = []
    test_classes = []
    n_train = 12

    train = open("train.txt", "w")
    test = open("test.txt", "w")
    ground = open("ground.txt", "w")

    test_counter = {c: 0 for c in np.unique(classes)}
    shuffle_indices = np.random.permutation(len(files))

    for s in shuffle_indices:
        if test_counter[classes[s]] < n_train:
            test_files.append(files[s])
            test_classes.append(classes[s])
            test_counter[classes[s]] += 1
        else:
            train_files.append(files[s])
            train_classes.append(classes[s])

    for i, f in enumerate(train_files):
        filename = os.path.join(data_path, f)
        train.write("{} {}\n".format(filename, train_classes[i]))
    train.close()

    for f in test_files:
        filename = os.path.join(data_path, f)
        test.write("{}\n".format(filename))
    test.close()

    for f in test_classes:
        ground.write("{}\n".format(f))
    ground.close()
