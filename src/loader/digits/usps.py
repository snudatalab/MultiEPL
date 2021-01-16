import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import random

data_dir = '../../data/digits'
file_name = 'usps_28x28.mat'


def load_usps(root=data_dir, data_num=-1, file_name=file_name):
    data_file_name = os.path.join(root, file_name)
    usps_data = sio.loadmat(data_file_name)

    usps_train = usps_data['dataset'][0][0].transpose(0, 2, 3, 1)
    usps_train = np.concatenate([usps_train, usps_train, usps_train], axis=3)
    usps_train = (usps_train * 255).astype(np.uint8)
    usps_labels_train = usps_data['dataset'][0][1].reshape(-1)

    usps_test = usps_data['dataset'][1][0].transpose(0, 2, 3, 1)
    usps_test = np.concatenate([usps_test, usps_test, usps_test], axis=3)
    usps_test = (usps_test * 255).astype(np.uint8)
    test_label = usps_data['dataset'][1][1].reshape(-1)

    inds = np.random.permutation(usps_train.shape[0])
    usps_train = usps_train[inds]
    train_label = usps_labels_train[inds]

    if 0 <= data_num:
        usps_train = usps_train[:data_num]
        train_label = train_label[:data_num]

    data_per_label = {}
    for label in range(10):
        inds = np.where(train_label == label)
        usps_train_label = usps_train[inds]
        data_per_label[label] = usps_train_label

    assert sum([len(data_per_label[key]) for key in data_per_label.keys()]) == usps_train.shape[0]

    print('*** USPS DATASET ***')
    print('Training data: {}, Training label: {}'.format(usps_train.shape, train_label.shape))
    print('Test data: {}, Test label: {}'.format(usps_test.shape, test_label.shape))

    return usps_train, usps_test, train_label, test_label, data_per_label


if __name__ == '__main__':
    data_train, data_test, label_train, label_test, data_label_dict = load_usps(data_num=500)
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

    plt.imshow(data_label_dict[1][0])
    plt.show()
    plt.imshow(data_label_dict[1][1])
    plt.show()

    length = 0
    for key in data_label_dict.keys():
        length += len(data_label_dict[key])
    print(length)