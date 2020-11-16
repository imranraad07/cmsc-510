# /**
#  * @author Mia Mohammad Imran
#  */

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


class Model(Model):
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
        out = tf.reshape(x, [-1, 28, 28, 1])
        out = self.conv1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.out(out)
        return out


cnn = Model()
optimizer = tf.optimizers.Adam(learning_rate)

time0 = time()
epochs = 15

cnn.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=["accuracy"],
)

history = cnn.fit(x_train, y_train, batch_size=batch_size, epochs=15)

print("Training Time (in minutes) =", (time() - time0) / 60)

plt.plot(history.epoch, history.history['loss'], color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Loss rate vs. iterations of training dataset (tensorflow)')
plt.show()

pred = cnn.fit((x_test), y_test)
print("Model Accuracy = ", pred.history['accuracy'])
