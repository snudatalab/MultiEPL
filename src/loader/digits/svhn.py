"""
Multi-EPL: Accurate Multi-Source Domain Adaptation

Authors:
- Seongmin Lee (ligi214@snu.ac.kr)
- Hyunsik Jeon (jeon185@gmail.com)
- U Kang (ukang@snu.ac.kr)

File: src/loader/digits/svhn.py
- Contains source code for setting SVHN train and test data
"""

import scipy.io as sio
import numpy as np
import os

data_dir = '../../data/digits'
train_file_name = 'svhn_train_32x32.mat'
test_file_name = 'svhn_test_32x32.mat'


def load_svhn(root=data_dir, data_num=-1, train_file_name=train_file_name, test_file_name=test_file_name):
    """
    Load SVHN training and test data
    :param root: the directory where Digits-Five data are saved
    :param data_num: the number of training data unless negative; otherwise, whole data are used as training data)
    :param train_file_name: the name of SVHN training data file
    :param test_file_name: the name of SVHN test data file
    :return: Training images, Test images, Training labels, Test labels
    """
    train_data_file_name = os.path.join(root, train_file_name)
    test_data_file_name = os.path.join(root, test_file_name)
    svhn_train_data = sio.loadmat(train_data_file_name)
    svhn_test_data = sio.loadmat(test_data_file_name)

    svhn_train = svhn_train_data['X'].transpose(3, 0, 1, 2).astype(np.uint8)
    train_label = svhn_train_data['y'].reshape(-1) % 10
    svhn_test = svhn_test_data['X'].transpose(3, 0, 1, 2).astype(np.uint8)
    test_label = svhn_test_data['y'].reshape(-1) % 10

    inds = np.random.permutation(svhn_train.shape[0])
    svhn_train = svhn_train[inds]
    train_label = train_label[inds]

    if 0 <= data_num:
        svhn_train = svhn_train[:data_num]
        train_label = train_label[:data_num]

    print('*** SVHN DATASET ***')
    print('Training data: {}, Training label: {}'.format(svhn_train.shape, train_label.shape))
    print('Test data: {}, Test label: {}'.format(svhn_test.shape, test_label.shape))

    return svhn_train, svhn_test, train_label, test_label
