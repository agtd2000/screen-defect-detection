#!/bin/bash

source activate
conda create -n phone -y
conda activate phone
conda install -y ipython

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install torch torchvision cython opencv-python notebook
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

git clone https://github.com/rewqazxv/maskscoring_rcnn.detectron2.git maskscoring_rcnn
cd maskscoring_rcnn && pip install -e .

mkdir -p pre_train_model
cd pre_train_model
wget -4 -O R-50.pkl https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
