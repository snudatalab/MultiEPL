"""
Multi-EPL: Accurate Multi-Source Domain Adaptation

Authors:
- Seongmin Lee (ligi214@snu.ac.kr)
- Hyunsik Jeon (jeon185@gmail.com)
- U Kang (ukang@snu.ac.kr)

File: src/loader/dataloader.py
- Contains source code for setting Digits-Five dataset and dataloader
"""

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from loader.digits.mnist import load_mnist
from loader.digits.mnist_m import load_mnist_m
from loader.digits.svhn import load_svhn
from loader.digits.synthdigits import load_synthdigits
from loader.digits.usps import load_usps

digits_data_dir = '../../data/digits'

digits_transform = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])


class GeneralDataset(Dataset):
    """ General dataset for Digits-Five """
    def __init__(self, images, labels, transform, target_train):
        super(GeneralDataset, self).__init__()
        self.images = images
        if target_train:
            self.labels = -np.ones_like(labels)
        else:
            self.labels = labels
        self.num_data = len(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label, 'index': idx}
        return sample


def get_digits_dataloader(name='MNIST', target=False, transform=digits_transform, batch_size=64, data_num=-1, data_dir=digits_data_dir):
    """
    Get dataset and dataloader for a Digits-Five dataset
    :param name: name of the dataset (one of MNIST, MNIST-M, SVHN, SYN, and USPS)
    :param target: True if the requested dataset is target dataset
    :param transform: how to transform images
    :param batch_size: batch size for dataloader
    :param data_num: the number of training data
    :param data_dir: the directory where Digits-Five data are saved
    :return: Training dataset, Training dataloader, Test dataset, Test dataloader for the requested dataset
    """
    if name == 'MNIST':
        print('Load MNIST data')
        train_data, test_data, train_label, test_label = load_mnist(data_dir, data_num)
    elif name == 'MNIST-M':
        print('Load MNIST-M data')
        train_data, test_data, train_label, test_label = load_mnist_m(data_dir, data_num)
    elif name == 'SVHN':
        print('Load SVHN data')
        train_data, test_data, train_label, test_label = load_svhn(data_dir, data_num)
    elif name == 'SYN':
        print('Load SYN data')
        train_data, test_data, train_label, test_label = load_synthdigits(data_dir, data_num)
    elif name == 'USPS':
        print('Load USPS data')
        train_data, test_data, train_label, test_label = load_usps(data_dir, data_num)
    else:
        raise ValueError('Name should be one of MNIST, MNIST-M, SVHN, SYN, and USPS')

    train_dataset = GeneralDataset(train_data, train_label, transform, target)
    test_dataset = GeneralDataset(test_data, test_label, transform, False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_dataset, train_dataloader, test_dataset, test_dataloader
