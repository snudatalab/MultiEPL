Multi-EPL

This package provides implementations of Multi-EPL, which is submitted at PLOS ONE.

## Overview
#### Code structure
```
MultiEPL
├── data
│   └── digits: Digits-Five dataset
├── script
│   └── digits.sh: Shell file for Digits-Five experiments
└── src
    ├── loader
    │   ├── digits: for Digits-Five dataset setting
    │   │   ├── mnist.py
    │   │   ├── mnist_m.py
    │   │   ├── svhn.py
    │   │   ├── synthdigits.py
    │   │   └── usps.py
    │   └── dataloader.py: generate dataloader and dataset of Digits-Five
    ├── utils
    │   └── default_param.py: set default parameters
    ├── network
    │   └── network_digits.py: network for Multi-EPL with Digits-Five dataset
    ├── solver.py: solver class for training
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

## Install
#### Environment
* Ubuntu
* CUDA 10.0
* Python 3.6.12
* torch 1.7.1
* torchvision 0.8.2
* scipy 1.5.4

To create an anaconda environment with  all the requirements:
``` shell
conda env create -n <ENV_NAME> -f requirement.txt
```

## How to use
``` shell
git clone https://github.com/snudatalab/AUBER.git
cd Multi-EPL/script/
sh digits.sh
```

## Contact us
* Seongmin Lee (ligi214@snu.ac.kr)
* Hyunsik Jeon (jeon185@gmail.com)
* U Kang (ukang@snu.ac.kr)
* Data Mining Lab at Seoul National University.
