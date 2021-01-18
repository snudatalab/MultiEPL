"""
Multi-EPL: Accurate Multi-Source Domain Adaptation

Authors:
- Seongmin Lee (ligi214@snu.ac.kr)
- Hyunsik Jeon (jeon185@gmail.com)
- U Kang (ukang@snu.ac.kr)

File: src/digits.py
- Contains source code for the main code for the Digits-Five experiments
"""

import torch
import numpy as np
import random

from utils.default_param import get_default_param_digits
from loader.dataloader import get_digits_dataloader
from solver import SolverDigits

source = ["MNIST", "MNIST-M", "SVHN", "SYN", "USPS"]


def main(args):
    target = args.target
    batch_size = args.batch_size
    target_data_num = args.target_data_num
    source_data_num = args.source_data_num
    data_dir = args.data_dir
    source.remove(target)
    if len(source) != 4:
        raise ValueError("Wrong input for target domain")

    # get target dataloader
    target_train_dataset, target_train_dataloader, target_test_dataset, target_test_dataloader = \
        get_digits_dataloader(name=target, target=True, batch_size=batch_size, data_num=target_data_num, data_dir=data_dir)

    # get source dataloader
    source_train_dataset_1, source_train_dataloader_1, source_test_dataset_1, source_test_dataloader_1 = \
        get_digits_dataloader(name=source[0], target=False, batch_size=batch_size, data_num=source_data_num, data_dir=data_dir)
    source_train_dataset_2, source_train_dataloader_2, source_test_dataset_2, source_test_dataloader_2 = \
        get_digits_dataloader(name=source[1], target=False, batch_size=batch_size, data_num=source_data_num, data_dir=data_dir)
    source_train_dataset_3, source_train_dataloader_3, source_test_dataset_3, source_test_dataloader_3 = \
        get_digits_dataloader(name=source[2], target=False, batch_size=batch_size, data_num=source_data_num, data_dir=data_dir)
    source_train_dataset_4, source_train_dataloader_4, source_test_dataset_4, source_test_dataloader_4 = \
        get_digits_dataloader(name=source[3], target=False, batch_size=batch_size, data_num=source_data_num, data_dir=data_dir)
    source_train_dataloader = [source_train_dataloader_1, source_train_dataloader_2, source_train_dataloader_3, source_train_dataloader_4]
    source_test_dataloader = [source_test_dataloader_1, source_test_dataloader_2, source_test_dataloader_3, source_test_dataloader_4]

    # set solver for model training
    solver = SolverDigits(args, target, source, target_train_dataset, target_train_dataloader, target_test_dataloader,
                          source_train_dataloader, source_test_dataloader)

    # Model training and test
    epochs = args.epochs
    max_acc = 0
    for e in range(1, epochs+1):
        solver.train(e)
        acc, loss = solver.test()
        if acc > max_acc:
            max_acc = acc
        print('Epoch {:03d} Test --- Accuracy: {:08f}, Loss: {:06f}'.format(e, acc, loss))
        print('MultiEPL-{} with target {} Epoch {:03d} Max Accuracy: {:08f}'.format(args.ensemble_num, target, e, max_acc))


if __name__ == '__main__':
    parser = get_default_param_digits()
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    main(args)
