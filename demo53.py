import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(mean=[0, 3], cov=[[1., 0.5], [0.5, 1.]],
                                                 size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(mean=[-3, 0], cov=[[1., 0.5], [0.5, 1]],
                                                 size=num_samples_per_class)
inputs = np.vstack((negative_samples, positive_samples)).astype('float32')
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype=float),
                     np.ones((num_samples_per_class, 1), dtype=float)))
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
# plt.show()

input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


def model(inputs):
    return tf.matmul(inputs, W) + b


def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)


learning_rate = 0.01


def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(predictions, targets)
    grad_loss_w, grad_loss_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_w * learning_rate)
    b.assign_sub(grad_loss_b * learning_rate)
    return loss


for step in range(400):
    loss = training_step(inputs, targets)
    print("loss at step {} is {:.4f}".format(step, loss))
plt.figure()
predictions = model(inputs)
x = np.linspace(-6, 6, 100)
y2 = -W[0] / W[1] * x + (0 - b) / W[1]
y1 = -W[0] / W[1] * x + (0.5 - b) / W[1]
y3 = -W[0] / W[1] * x + (1 - b) / W[1]
plt.plot(x, y1, '-r')
plt.plot(x, y2, '--b')
plt.plot(x, y3, '--b')
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()