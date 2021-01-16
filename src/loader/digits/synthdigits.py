"""
Multi-EPL: Accurate Multi-Source Domain Adaptation

Authors:
- Seongmin Lee (ligi214@snu.ac.kr)
- Hyunsik Jeon (jeon185@gmail.com)
- U Kang (ukang@snu.ac.kr)

File: src/loader/digits/synthdigits.py
- Contains source code for setting SynthDigits train and test data
"""

import scipy.io as sio
import numpy as np
import os

data_dir = '../../data/digits'
file_name = 'syn_number.mat'


def load_synthdigits(root=data_dir, data_num=-1, file_name=file_name):
    """
    Load SynthDigits training and test data
    :param root: the directory where Digits-Five data are saved
    :param data_num: the number of training data unless negative; otherwise, whole data are used as training data)
    :param file_name: the name of SynthDigits data file
    :return: Training images, Test images, Training labels, Test labels
    """
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

    print('*** SYN DATASET ***')
    print('Training data: {}, Training label: {}'.format(syn_train.shape, train_label.shape))
    print('Test data: {}, Test label: {}'.format(syn_test.shape, test_label.shape))

    return syn_train, syn_test, train_label, test_label
