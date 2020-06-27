#!/bin/bash

source activate phone
python train.py --eval-only MODEL.WEIGHTS output/model_final.pth
