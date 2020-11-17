# /**
#  * @author Mia Mohammad Imran
#  */
from abc import ABC
from time import time
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow import keras

batch_size = 64
hidden_sizes = [32, 16]
output_size = 10
learning_rate = 1e-3

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train / 255.0, x_test / 255.0

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


class Model(Model, ABC):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = layers.Conv2D(hidden_sizes[0], kernel_size=5, activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv2 = layers.Conv2D(hidden_sizes[1], kernel_size=5, activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024)
        self.dropout = layers.Dropout(rate=0.5)
        self.out = layers.Dense(output_size)

    def call(self, x, **kwargs):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.out(x)
        return x


cnn = Model()
optimizer = tf.optimizers.Adam(learning_rate)

time0 = time()
epochs = 15

cnn.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=["accuracy"],
)

history = cnn.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))


class ModelBN(Model, ABC):
    def __init__(self):
        super(ModelBN, self).__init__()
        self.conv1 = layers.Conv2D(hidden_sizes[0], kernel_size=5, activation=tf.nn.relu)
        self.bn1 = layers.BatchNormalization()
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv2 = layers.Conv2D(hidden_sizes[1], kernel_size=5, activation=tf.nn.relu)
        self.bn2 = layers.BatchNormalization()
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024)
        self.bn3 = layers.BatchNormalization()
        self.dropout = layers.Dropout(rate=0.5)
        self.out = layers.Dense(output_size)
        self.bn4 = layers.BatchNormalization()

    def call(self, x, **kwargs):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.bn4(x)
        return x


cnnBN = ModelBN()
optimizerBN = tf.optimizers.Adam(learning_rate)

time0 = time()

cnnBN.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizerBN,
    metrics=["accuracy"],
)

historyBN = cnnBN.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

print("Training Time (in minutes) =", (time() - time0) / 60)
plt.plot(historyBN.epoch, historyBN.history['loss'], color='red', label='Batch Norm')
plt.plot(history.epoch, history.history['loss'], color='blue', label='Without Batch Norm')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Loss rate vs. iterations of training dataset (tensorflow)')
plt.show()

# 0.9909
# 0.9909
