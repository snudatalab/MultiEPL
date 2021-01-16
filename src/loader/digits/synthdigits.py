import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import random

data_dir = '../../data/digits'
file_name = 'syn_number.mat'


def load_synthdigits(root=data_dir, data_num=-1, file_name=file_name):
    data_file_name = os.path.join(root, file_name)
    syn_data = sio.loadmat(data_file_name)

    syn_train = syn_data['train_data'].astype(np.uint8)
    train_label = syn_data['train_label'].reshape(-1)
    syn_test = syn_data['test_data'].astype(np.uint8)
    test_label = syn_data['test_label'].reshape(-1)

    inds = np.random.permutation(syn_train.shape[0])
    syn_train = syn_train[inds]
    train_label = train_label[inds]

    if 0 <= data_num:
        syn_train = syn_train[:data_num]
        train_label = train_label[:data_num]

    data_per_label = {}
    for label in range(10):
        inds = np.where(train_label == label)
        svhn_train_label = syn_train[inds]
        data_per_label[label] = svhn_train_label

    assert sum([len(data_per_label[key]) for key in data_per_label.keys()]) == syn_train.shape[0]

    print('*** SYN DATASET ***')
    print('Training data: {}, Training label: {}'.format(syn_train.shape, train_label.shape))
    print('Test data: {}, Test label: {}'.format(syn_test.shape, test_label.shape))

    return syn_train, syn_test, train_label, test_label, data_per_label


if __name__ == '__main__':
    data_train, data_test, label_train, label_test, data_label_dict = load_synthdigits(data_num=500)
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