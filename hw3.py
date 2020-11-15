# /**
#  * @author Mia Mohammad Imran
#  */


import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from keras.datasets import mnist

tf.disable_v2_behavior()
c0 = 1
c1 = 7
threshold = 1e-3
learn_rate = 1.0
n_epochs = 100
input_dim = 784
batch_size = 128


def preprocess(dataset_x, dataset_y):
    dataset_x_new = []
    dataset_y_new = []
    for i in range(len(dataset_y)):
        if dataset_y[i] == c0 or dataset_y[i] == c1:
            dataset_x_new.append(dataset_x[i])
            if dataset_y[i] == c0:
                dataset_y_new.append(1)
            elif dataset_y[i] == c1:
                dataset_y_new.append(-1)
    dataset_y_new = np.array(dataset_y_new).reshape(len(dataset_y_new), 1)
    dataset_x_new = np.reshape(dataset_x_new, (len(dataset_x_new), input_dim))
    dataset_x_new = [[x for x in sample] for sample in dataset_x_new]
    return dataset_x_new, dataset_y_new


def train(x_train, y_train):
    n_samples, n_features = x_train.shape

    w = tf.Variable(np.random.rand(input_dim, 1).astype(dtype='float32'), name="weight")
    b = tf.Variable(0.0, dtype=tf.float32, name="bias")

    x = tf.placeholder(dtype=tf.float32, name='x')
    y = tf.placeholder(dtype=tf.float32, name='y')

    predictions = tf.matmul(x, w) + b
    loss = tf.reduce_mean(tf.log(1 + tf.exp(tf.multiply(-1.0 * y, predictions))))

    # optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
    optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learn_rate,
                                                          l1_regularization_strength=0.1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            for idx in range(0, n_samples, batch_size):
                iE = min(n_samples, idx + batch_size)
                x_batch = x_train[idx:iE, :]
                y_batch = y_train[idx:iE, :]
                sess.run([optimizer], feed_dict={x: x_batch, y: y_batch})
            curr_w, curr_b = sess.run([w, b])

            for idx in range(len(curr_w)):
                if curr_w[idx] < threshold * -1:
                    curr_w[idx] += threshold
                else:
                    curr_w[idx] -= threshold
            sess.run([tf.assign(w, curr_w)])
    return curr_w, curr_b


def predict(w, b, x_test):
    labels = []
    for item in x_test:
        item = np.rad2deg(item)
        u = np.matmul(item, w) + b
        if u < 0:
            labels.append(-1)
        else:
            labels.append(1)
    return labels


def evaluate(labels, original):
    labels = list(labels)
    agreed = 0
    for i in range(len(labels)):
        if labels[i] == original[i]:
            agreed += 1
    return agreed / len(original)


training_size = []
accuracy_percentage = []


def main(sample_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    sample_indexes = random.sample(range(len(x_train)), int(len(x_train) * sample_size))

    x_train_sample = np.array([_ for i, _ in enumerate(x_train) if i in sample_indexes])
    y_train_sample = np.array([_ for i, _ in enumerate(y_train) if i in sample_indexes])

    w, b = train(x_train_sample, y_train_sample)
    labels = predict(w, b, x_test)

    accuracy = evaluate(labels, y_test)

    training_size.append(len(sample_indexes))
    accuracy_percentage.append(accuracy)
    print(accuracy, len(sample_indexes))


if __name__ == '__main__':
    main(1)
    main(.9)
    main(.75)
    main(.50)
    main(.25)
    main(.1)
    # print(training_size)
    # print(accuracy_percentage)

    plt.plot(training_size, accuracy_percentage, color='red')
    plt.xlim(training_size[0], 0)
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    title = 'Training data set size vs. accuracy'
    plt.title(title)
    plt.show()
