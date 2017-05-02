import cv2
import pdb
import matplotlib.pyplot as plt
import os
import numpy as np

data_dir = 'datasets/Training/'
folders = os.listdir(os.getcwd() + '/' + data_dir)
folders.remove("Readme.txt")
folders = [os.getcwd() + '/' + data_dir + x for x in folders]
images = list()

for folder in folders:
    image = [x for x in os.listdir(folder) if 'ppm' in x][0]
    img = cv2.imread(folder + '/' + image)
    images.append(img)

for i in range(62):
    plt.subplot(11,6,i+1)
    plt.imshow(images[i])
    plt.axis('off')
plt.show()
