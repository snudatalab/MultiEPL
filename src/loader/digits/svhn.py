import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import random

data_dir = '../../data/digits'
train_file_name = 'svhn_train_32x32.mat'
test_file_name = 'svhn_test_32x32.mat'


def load_svhn(root=data_dir, data_num=-1, train_file_name=train_file_name, test_file_name=test_file_name):
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

    data_per_label = {}
    for label in range(10):
        inds = np.where(train_label == label)
        svhn_train_label = svhn_train[inds]
        data_per_label[label] = svhn_train_label

    assert sum([len(data_per_label[key]) for key in data_per_label.keys()]) == svhn_train.shape[0]

    print('*** SVHN DATASET ***')
    print('Training data: {}, Training label: {}'.format(svhn_train.shape, train_label.shape))
    print('Test data: {}, Test label: {}'.format(svhn_test.shape, test_label.shape))

    return svhn_train, svhn_test, train_label, test_label, data_per_label


if __name__ == '__main__':
    data_train, data_test, label_train, label_test, data_label_dict = load_svhn(data_num=500)
    print(data_train.shape)
    print(data_test.shape)
    print(label_train.shape)
    print(label_test.shape)
    print(data_label_dict.keys())
    print(np.max(data_train))
    for i in range(3):
        plt.imshow(data_train[i])
        plt.show()
        print(label_train[i])

    plt.imshow(data_label_dict[0][0])
    plt.show()
    plt.imshow(data_label_dict[0][1])
    plt.show()