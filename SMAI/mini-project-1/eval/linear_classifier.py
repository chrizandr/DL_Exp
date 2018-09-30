import sys
import random

train_file = sys.argv[1]
test_file = sys.argv[2]

with open(train_file, "r") as f:
    lines = f.readlines()

label_set = set()
for l in lines:
    img_path, label = l.strip().split()
    label_set.add(label)

# train a classifier

with open(test_file, "r") as f:
    test_paths = f.readlines()

for p in test_paths:
    # predict the label for image p
    print(random.sample(label_set, 1)[0])
