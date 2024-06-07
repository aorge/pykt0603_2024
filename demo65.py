from matplotlib import pyplot as plt
import tensorflow as tf
from keras import datasets

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
print(train_images.shape, test_images.shape)
print(train_labels.shape, test_labels.shape)


def plotImage(i):
    plt.title('image marked as:%d' % train_labels[i])
    plt.imshow(train_images[i], cmap='binary')
    plt.show()


def plotTestImage(i):
    plt.title('TEST image marked as:%d' % test_labels[i])
    plt.imshow(test_images[i], cmap='binary')
    plt.show()

#plotImage(205)

plotTestImage(200)