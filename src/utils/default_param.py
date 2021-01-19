"""
Multi-EPL: Accurate Multi-Source Domain Adaptation

Authors:
- Seongmin Lee (ligi214@snu.ac.kr)
- Hyunsik Jeon (jeon185@gmail.com)
- U Kang (ukang@snu.ac.kr)

File: src/utils/default_param.py
- Contains source code for the input arguments for each experiment
"""

import argparse


def get_default_param_digits():
    """
    Receive arguments
    :return: Argument parser
    """
    parser = argparse.ArgumentParser()

    # Domain Adaptation Options for Digits-Five Dataset
    parser.add_argument("--target",
                        help="target domain",
                        choices=["MNIST", "MNIST-M", "SVHN", "SYN", "USPS"],
                        type=str,
                        default="MNIST")
    parser.add_argument("--source_data_num",
                        help="number of training source data, -1 means using all the given data",
                        type=int,
                        default=25000)
    parser.add_argument("--target_data_num",
                        help="number of training source data, -1 means using all the given data",
                        type=int,
                        default=9000)
    parser.add_argument("--num_classes",
                        help="number of classes in the dataset",
                        type=int,
                        default=10)
    parser.add_argument("--input_size",
                        help="size of the input images",
                        type=int,
                        default=32)
    parser.add_argument("--ensemble_num",
                        help="the number of network pairs",
                        choices=[1, 2], type=int,
                        default=1)
    parser.add_argument("--pseudolabel_setting_interval",
                        help="pseudolabel setting interval",
                        type=float, default=10)

    # Model Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=5e-4)
    parser.add_argument("--g_loss_weight", type=float, default=5e-4)
    parser.add_argument("--lc_loss_weight", type=float, default=1)

    # Directory to save results and logs
    parser.add_argument("--data_dir", help="path where the digits data are saved", type=str)

    # Miscellaneous
    parser.add_argument("--gpu", help="gpu id", type=int, default=0)
    parser.add_argument("--seed", help="random seed", type=int, default=0)

    return parser
