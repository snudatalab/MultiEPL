"""
Multi-EPL: Accurate Multi-Source Domain Adaptation

Authors:
- Seongmin Lee (ligi214@snu.ac.kr)
- Hyunsik Jeon (jeon185@gmail.com)
- U Kang (ukang@snu.ac.kr)

File: src/loader/digits/mnist.py
- Contains source code for setting MNIST train and test data
"""

import scipy.io as sio
import numpy as np
import os

data_dir = '../../../data/digits'
file_name = 'mnist_data.mat'


def load_mnist(root=data_dir, data_num=-1, file_name=file_name):
    """
    Load MNIST training and test data
    :param root: the directory where Digits-Five data are saved
    :param data_num: the number of training data unless negative; otherwise, whole data are used as training data)
    :param file_name: the name of MNIST data file
    :return: Training images, Test images, Training labels, Test labels
    """
    data_file_name = os.path.join(root, file_name)
    mnist_data = sio.loadmat(data_file_name)
    mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3).astype(np.uint8)
    mnist_labels_train = mnist_data['label_train']

    mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3).astype(np.uint8)
    mnist_labels_test = mnist_data['label_test']

    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]

    test_label = np.argmax(mnist_labels_test, axis=1)

    if 0 <= data_num:
        mnist_train = mnist_train[:data_num]
        train_label = train_label[:data_num]

    print('*** MNIST DATASET ***')
    print('Training data: {}, Training label: {}'.format(mnist_train.shape, train_label.shape))
    print('Test data: {}, Test label: {}'.format(mnist_test.shape, test_label.shape))

    return mnist_train, mnist_test, train_label, test_label
