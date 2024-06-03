import tensorflow as tf
import numpy as np
import pandas as pd

from typing import List, Tuple
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split
# from sklearn.inspection import DecisionBoundaryDisplay <- doesn't work because we have 10 dimensions... or at least,
# I haven't figured out how to make it work with those

def printer():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    import matplotlib.pyplot as plt

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print('Example training images and their labels: ' + str([x[0] for x in y_train[0:10]]))
    print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[0:5]]))

    f, axarr = plt.subplots(1, 10)
    f.set_size_inches(16, 6)

    for i in range(10):
        img = x_train[i]
        axarr[i].imshow(img)
    plt.show()






    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    y_train_flat = y_train.ravel()
    y_test_flat = y_test.ravel()

    figure, axes = plt.subplots(1, figsize=(8, 8))

    # k_val = [1, 10, 20, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 7500, 10000, 15000]
    k_val = [10]

    train_acc = []
    test_acc = []

    for i, k in enumerate(k_val):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train_flat, y_train_flat)

        y_pred_train = knn.predict(x_train_flat)
        y_pred_test = knn.predict(x_test_flat)

        train = 1 - accuracy_score(y_train_flat, y_pred_train)
        test = 1 - accuracy_score(y_test_flat, y_pred_test)

        train_acc.append(train)
        test_acc.append(test)

    plt.semilogx(k_val, train_acc, c='red', label='Training')
    plt.semilogx(k_val, test_acc, c='green', label='Testing')
    plt.legend()

    plt.set_xlabel('k')
    plt.set_ylabel('Error Rate')



# kNN Classifier
# For now we are using the default test/train split just to verify a proof of concept that it works

# This first part was taken from HW #1's 3.1 kNN neighbors graphing question
# okay wait this won't work we have. 10 dimensions. Idk how you're supposed to graph that, even...
# label_to_int = {1: 'red', 2: 'orange', 2: 'yellow', 4: 'green', 5: 'blue', 6: 'indigo', 7: 'violet', 8: 'white', 9: 'black', 10: 'pink'}
# print(type(y_train))
# category = y_train.map(label_to_int)
# nevermind we have to subset it. This didn't finish running last night oops

#plot_kwargs = {'cmap': 'viridis',
#               'response_method': 'predict',
#               'plot_method': 'pcolormesh',
#               'shading': 'auto',
#               'alpha': 0.5,
#               'grid_resolution': 100}

# need to flatten it so it works with knn neighbors)
