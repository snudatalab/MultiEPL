"""
Multi-EPL: Accurate Multi-Source Domain Adaptation

Authors:
- Seongmin Lee (ligi214@snu.ac.kr)
- Hyunsik Jeon (jeon185@gmail.com)
- U Kang (ukang@snu.ac.kr)

File: src/loader/digits/usps.py
- Contains source code for setting USPS train and test data
"""

import scipy.io as sio
import numpy as np
import os

data_dir = '../../data/digits'
file_name = 'usps_28x28.mat'


def load_usps(root=data_dir, data_num=-1, file_name=file_name):
    """
    Load USPS training and test data
    :param root: the directory where Digits-Five data are saved
    :param data_num: the number of training data unless negative; otherwise, whole data are used as training data)
    :param file_name: the name of USPS data file
    :return: Training images, Test images, Training labels, Test labels
    """
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

    print('*** USPS DATASET ***')
    print('Training data: {}, Training label: {}'.format(usps_train.shape, train_label.shape))
    print('Test data: {}, Test label: {}'.format(usps_test.shape, test_label.shape))

    return usps_train, usps_test, train_label, test_label
