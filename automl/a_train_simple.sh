#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -jc gs-container_g8
#$ -ac d=aip-tensorflow-2012-opencv-1
#$ -e /home/toshi/fake_detection/automl/projects_simple/EfficientNetB7_20230802-150615_epoch50/train_simple.sh.e$JOB_ID
#$ -o /home/toshi/fake_detection/automl/projects_simple/EfficientNetB7_20230802-150615_epoch50/train_simple.sh.o$JOB_ID


. /fefs/opt/dgx/env_set/nvcr-tensorflow-2012-tf2-py3.sh

. ~/proxy.sh
export TF_FORCE_GPU_ALLOW_GROWTH=true

/usr/bin/python -m pip install --upgrade pip
/usr/bin/python -m pip install matplotlib
/usr/bin/python -m pip install seaborn
/usr/bin/python -m pip install python-dotenv
/usr/bin/python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 torchmetrics==0.11.4 torchsummary==1.5.1 --extra-index-url https://download.pytorch.org/whl/cu116
/usr/bin/python -m pip install nni pytorch_lightning==2.0.2
/usr/bin/python -m pip install grad-cam
/usr/bin/python -m pip freeze

/usr/bin/python train_simple.py EfficientNetB7_20230802-150615_epoch50
