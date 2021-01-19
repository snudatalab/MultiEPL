#!/bin/bash

TARGET="MNIST-M"
SOURCE_DATA_NUM=25000
TARGET_DATA_NUM=9000
ENSEM_NUM=2

cd ..
WORK_DIR=$(pwd)
DATA_DIR="${WORK_DIR}/data/digits"

cd src

python digits.py \
  --target $TARGET \
  --source_data_num $SOURCE_DATA_NUM \
  --target_data_num $TARGET_DATA_NUM \
  --ensemble_num $ENSEM_NUM \
  --data_dir $DATA_DIR \
  --gpu 0 \
  --seed 0