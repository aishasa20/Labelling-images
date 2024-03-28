import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils


def main():

    cifar_data = tf.keras.datasets.cifar100.load_data(label_mode="fine")

    (X_train, y_train), (X_test, y_test) = cifar_data


    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # visualize the first 30 images in the training dataset
    class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                   'bowl',
                   'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
                   'chimpanzee', 'clock',
                   'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
                   'flatfish', 'forest',
                   'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                   'lizard',
                   'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
                   'orchid',
                   'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                   'possum',
                   'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
                   'skyscraper',
                   'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
                   'telephone',
                   'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
                   'willow_tree', 'wolf',
                   'woman', 'worm']

    # Visualize the first 30 images
    plt.figure(figsize=[10, 10])
    for i in range(30):
        plt.subplot(6, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i])
        plt.xlabel(class_names[y_train[i, 0]])
    plt.show()

    # scale the data
    X_train = X_train / 255
    X_test = X_test / 255

    # one-hot encoding the classes to use the categorical cross-entropy loss function
    n_classes = 100
    print("Shape before one-hot encoding:", y_train.shape)
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)
    print("Shape after one-hot encoding: ", y_train.shape)

    # Build a stack of layers with the sequential model from keras
    model = Sequential()
    # convolutional layer
    model.add(Conv2D(100, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", input_shape=(32, 32, 3)))
    # drop out layer
    model.add(Dropout(0.25))
    # convolutional layer
    model.add(Conv2D(100, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", input_shape=(32, 32, 3)))
    # max pool layer
    model.add(MaxPool2D(pool_size=(2, 2)))
    # dense layer
    model.add(Dense(250, activation="relu"))
    # flatten layer
    model.add(Flatten())
    # dense layer
    model.add(Dense(100, activation="relu"))
    model.build()
    model.summary()

    # compile the sequential model
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

    # train the model for 50 epochs
    history = model.fit(X_train, y_train, batch_size=50, epochs=75, validation_data=(X_test, y_test))

    # Loss curve
    plt.figure(figsize=[6, 4])
    plt.plot(history.history['loss'], 'black', linewidth=2.0)
    plt.plot(history.history["val_loss"], 'green', linewidth=2.0)
    plt.legend(["Training Loss", "Validation Loss"], fontsize=14)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.title("Loss Curve", fontsize=12)

    # Accuracy curve
    plt.figure(figsize=[6, 4])
    plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
    plt.plot(history.history["val_accuracy"], 'green', linewidth=2.0)
    plt.legend(["Training Accuracy", "Validation Loss"], fontsize=14)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.title("Loss Curve", fontsize=12)

    # predict
    pred = model.predict(X_test)

    # convert the predictions into label index
    pred_classes = np.argmax(pred, axis=1)

    # plot the actual results with actual and predicted labels
    plt.figure(figsize=[10, 10])
    # Convert y_test to integer values
    y_test = y_test.astype(int)
    for i in range(30):
        plt.subplot(6, 5, i + 1).imshow(X_test[i])
        plt.subplot(6, 5, i + 1).set_title("True: %s \nPredict: %s" %
                                           (class_names[y_test[i, 0]],
                                            class_names[pred_classes[i]]))

        plt.subplot(6, 5, i + 1).axis('off')

    plt.subplots_adjust(hspace=1)
    plt.show()

    # visualize 30 misclassified images
    failed_indices = []
    idx = 0

    # Get list of all the failed indices
    for i in y_test:
        if i[0] != pred_classes[idx]:
            failed_indices.append(idx)
        idx = idx + 1

    # randomly select 30 failed indices
    random_indices = random.sample(failed_indices, 30)

    plt.figure(figsize=[10, 10])
    for i in range(30):
        plt.subplot(6, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(random_indices[i])
        plt.xlabel(class_names[y_train[i, 0]])
    plt.show()


if __name__ == '__main__':
    main()
