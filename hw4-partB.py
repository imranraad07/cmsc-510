# /**
#  * @author Mia Mohammad Imran
#  */

from time import time
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

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


def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)


def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = cnn(x)
        loss = cross_entropy_loss(pred, y)

    trainable_variables = cnn.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


training_steps = 938
time0 = time()
epochs = 15
losses = []
iteration = []
for epoch in range(epochs):
    running_loss = 0
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
        run_optimization(batch_x, batch_y)
        pred = cnn(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        running_loss = running_loss + loss
    print("Epoch {} - Training loss: {}".format(epoch, running_loss / training_steps))
    losses.append(running_loss)
    iteration.append(epoch)

print("Training Time (in minutes) =", (time() - time0) / 60)

plt.plot(iteration, losses, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Loss rate vs. iterations of training dataset (tensorflow)')
plt.show()

pred = cnn(x_test)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.cast(y_test, tf.int64))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

print("Model Accuracy = ", acc)

# Model Accuracy =  tf.Tensor(0.9899, shape=(), dtype=float32)
