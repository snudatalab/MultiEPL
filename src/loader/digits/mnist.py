import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = '../../../data/digits'
file_name = 'mnist_data.mat'


def load_mnist(root=data_dir, data_num=-1, file_name=file_name):
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

    data_per_label = {}
    for label in range(10):
        inds = np.where(train_label == label)
        mnist_train_label = mnist_train[inds]
        data_per_label[label] = mnist_train_label

    assert sum([len(data_per_label[key]) for key in data_per_label.keys()]) == mnist_train.shape[0]

    print('*** MNIST DATASET ***')
    print('Training data: {}, Training label: {}'.format(mnist_train.shape, train_label.shape))
    print('Test data: {}, Test label: {}'.format(mnist_test.shape, test_label.shape))

    return mnist_train, mnist_test, train_label, test_label, data_per_label


if __name__ == '__main__':
    data_train, data_test, label_train, label_test, data_label_dict = load_mnist(data_num=500)
    print(data_train.shape)
    print(data_test.shape)
    print(label_train.shape)
    print(label_test.shape)
    print(data_label_dict.keys())
    print(data_train)
    print(np.max(data_train))
    for i in range(3):
        plt.imshow(data_train[i])
        plt.show()
        print(label_train[i])