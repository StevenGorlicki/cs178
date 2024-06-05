from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from unpickle import unpickle_f
import tensorflow as tf
import numpy as np
import pandas as pd

from typing import List, Tuple
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from Printing_images import printer


def main():
    files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    dicts, xs, ys = unpickle_f(files)

    # Shuffle the data and select a small batch
    n_rows = len(ys)
    small_batch_size = 5000
    rows = np.random.choice(n_rows, small_batch_size, replace=False)

    image_selection = xs[rows]
    label_selection = ys[rows]

    x_train, x_test, y_train, y_test = train_test_split(image_selection, label_selection, test_size=0.25,
                                                        random_state=1234, shuffle=True)

    # Flatten the data
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    pca = PCA(n_components=100)  # Adjust n_components as needed
    x_train_flat = pca.fit_transform(x_train_flat)
    x_test_flat = pca.transform(x_test_flat)

    # Feature scaling
    scaler = StandardScaler()
    x_train_flat = scaler.fit_transform(x_train_flat)
    x_test_flat = scaler.transform(x_test_flat)

    y_train_flat = y_train.ravel()
    y_test_flat = y_test.ravel()

    # Train logistic regression model
    logistic_regression = LogisticRegression(max_iter=800, solver='saga',multi_class='multinomial')
    logistic_regression.fit(x_train_flat, y_train_flat)

    # Predict on training and test data
    y_pred_train = logistic_regression.predict(x_train_flat)
    y_pred_test = logistic_regression.predict(x_test_flat)

    # Calculate accuracy
    train_acc = accuracy_score(y_train_flat, y_pred_train)
    test_acc = accuracy_score(y_test_flat, y_pred_test)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_test_flat, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot()
    plt.show()

if __name__ == "__main__":
    main()
