
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD





def GradientDescent(X_train, X_valid, y_train, y_valid, batchSize: int):
    #Transform input to 1D arrays from 28 by 28 pixels grid
    X_train = X_train.reshape(60000, 784).astype('float32')
    X_valid = X_valid.reshape(10000, 784).astype('float32')

    #Coerse pixel values into a float between 1 and 0
    X_train /= 255
    X_valid /= 255

    #Output should be a classification of 10 possible digits (0-9)
    n_classes = 10
    y_train = to_categorical(y_train, n_classes)
    y_valid = to_categorical(y_valid, n_classes)

    

    #Create a feed forward network that has Dense node connections
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(784,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #Use CC as loss function and learning rate of 0.1
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

    return model.fit(X_train, y_train, batch_size=batchSize, epochs=20, verbose=1, validation_data=(X_valid, y_valid))

    

#mnist data set contains examples of handwritten digits
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

history = GradientDescent(X_train, X_valid, y_train, y_valid, 128)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.plot(history.history['accuracy'], label='accuracy')
plt.title('Gradient Descent types')
plt.ylabel('')
plt.xlabel('No. epoch')
plt.legend(loc=0)
plt.show()

