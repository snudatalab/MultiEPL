Multi-EPL

This package provides implementations of Multi-EPL, which is submitted at PLOS ONE.

## Overview
#### Code structure
```shell
Multi-EPL
├── data
│   └── digits: Digits-Five dataset
├── script
│   └── digits.sh
└── src
    ├── loader
    │   ├── dataloader.py: generate dataloader and dataset of Digits-Five
    │   └── digits
    │       ├── mnist_m.py
    │       ├── mnist.py
    │       ├── svhn.py
    │       ├── synthdigits.py
    │       └── usps.py
    ├── solver
    │   ├── solver_digits_test.py: solver class for demo experiments
    │   └── solver_digits.py: solver class for training
    ├── utils
    │   └── default_param.py: set default parameters
    ├── network
    │   └── network_digits.py: network for Multi-EPL with Digits-Five dataset
    ├── demo.py: coode for demo experiments
    └── digits.py: code for training Multi-EPL with Digits-Five dataset
```

#### Data description
* Digits-Five: Consists of five datasets for digit recognition
  * MNIST
  * MNIST-M
  * SVHN
  * SynthDigits
  * USPS
* Download link: [[data](https://drive.google.com/drive/folders/1MqeBt3SunyADs7gfAwd6U6ZlafRnKYYX?usp=sharing)]
* Add the data files at "Multi-EPL/data/digits" before running the codes
* Data source: https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit 
  * Note that we are unrelated to the group providing the data.

## Environment
``` shell
conda env create -n NAME -f requirement.txt
```

## How to use
``` shell
cd Multi-EPL/script/
sh train.sh
```
