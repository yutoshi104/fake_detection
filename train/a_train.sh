#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -jc gs-container_g8
#$ -ac d=aip-tensorflow-2012-opencv-1
#$ -e /home/toshi/fake_detection/model/Vgg16_20230319-095458_epoch300/retrain.sh.e$JOB_ID
#$ -o /home/toshi/fake_detection/model/Vgg16_20230319-095458_epoch300/retrain.sh.o$JOB_ID


. /fefs/opt/dgx/env_set/nvcr-tensorflow-2012-tf2-py3.sh

. ~/proxy.sh
export TF_FORCE_GPU_ALLOW_GROWTH=true

/usr/bin/python -m pip install --upgrade pip
/usr/bin/python -m pip install matplotlib
/usr/bin/python -m pip install python-dotenv
/usr/bin/python -m pip freeze

/usr/bin/python retrain.py Vgg16_20230319-095458_epoch300 16
