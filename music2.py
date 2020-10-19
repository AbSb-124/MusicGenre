import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "C:\\mfcc_data.json"


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def plot_history(history):
    
    fig, axs = plt.subplots(2)
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):

    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):

    
    musicmodel = keras.Sequential()

   
    musicmodel.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    musicmodel.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    musicmodel.add(keras.layers.BatchNormalization())

   
    musicmodel.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    musicmodel.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    musicmodel.add(keras.layers.BatchNormalization())

   
    musicmodel.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    musicmodel.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    musicmodel.add(keras.layers.BatchNormalization())

    
    musicmodel.add(keras.layers.Flatten())
    musicmodel.add(keras.layers.Dense(64, activation='relu'))
    musicmodel.add(keras.layers.Dropout(0.3))

   
    musicmodel.add(keras.layers.Dense(10, activation='softmax'))

    return musicmodel


def predict(model, X, y):

    
    X = X[np.newaxis, ...]

    
    prediction = model.predict(X)

    
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


if __name__ == "__main__":

    
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    
    ipshape= (X_train.shape[1], X_train.shape[2], 1)
    musicmodel = build_model(ipshape)

   
    opti = keras.optimizers.Adam(learning_rate=0.0001)
    musicmodel.compile(optimizer=opti,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    musicmodel.summary()

    hist = musicmodel.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

   
    plot_history(hist)

    
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTesting Accuracy:', acc)

   
    X_to_predict = X_test[100]
    y_to_predict = y_test[100]

    
    predict(musicmodel, X_to_predict, y_to_predict)
