"""
Multi-EPL: Accurate Multi-Source Domain Adaptation

Authors:
- Seongmin Lee (ligi214@snu.ac.kr)
- Hyunsik Jeon (jeon185@gmail.com)
- U Kang (ukang@snu.ac.kr)

File: src/loader/digits/mnist_m.py
- Contains source code for setting MNIST-M train and test data
"""

import scipy.io as sio
import numpy as np
import os

data_dir = '../../data/digits'
file_name = 'mnistm_with_label.mat'


def load_mnist_m(root=data_dir, data_num=-1, file_name=file_name):
    """
    Load MNIST-M training and test data
    :param root: the directory where Digits-Five data are saved
    :param data_num: the number of training data unless negative; otherwise, whole data are used as training data)
    :param file_name: the name of MNIST-M data file
    :return: Training images, Test images, Training labels, Test labels
    """
    data_file_name = os.path.join(root, file_name)
    mnist_m_data = sio.loadmat(data_file_name)

    mnist_m_train = mnist_m_data['train'].astype(np.uint8)
    mnist_m_labels_train = mnist_m_data['label_train']
    mnist_m_test = mnist_m_data['test'].astype(np.uint8)
    mnist_m_labels_test = mnist_m_data['label_test']

    train_label = np.argmax(mnist_m_labels_train, axis=1)
    inds = np.random.permutation(mnist_m_train.shape[0])
    mnist_m_train = mnist_m_train[inds]
    train_label = train_label[inds]

    test_label = np.argmax(mnist_m_labels_test, axis=1)

    if 0 <= data_num:
        mnist_m_train = mnist_m_train[:data_num]
        train_label = train_label[:data_num]

    print('*** MNIST-M DATASET ***')
    print('Training data: {}, Training label: {}'.format(mnist_m_train.shape, train_label.shape))
    print('Test data: {}, Test label: {}'.format(mnist_m_test.shape, test_label.shape))

    return mnist_m_train, mnist_m_test, train_label, test_label
