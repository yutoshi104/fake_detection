#!/bin/sh
#$ -S /bin/bash # UGE-unique line, sets interpreter shell as bash
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=aip-tensorflow-2012-opencv-1

# load proper environment variables
. /fefs/opt/dgx/env_set/nvcr-tensorflow-2012-tf2-py3.sh

# package install
. ~/proxy.sh

/usr/bin/python -m pip install --upgrade pip
/usr/bin/python -m pip install matplotlib
/usr/bin/python -m pip install seaborn
/usr/bin/python -m pip install python-dotenv
/usr/bin/python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 torchmetrics==0.11.4 torchsummary==1.5.1 --extra-index-url https://download.pytorch.org/whl/cu116
/usr/bin/python -m pip install nni pytorch_lightning==2.0.2
/usr/bin/python -m pip install grad-cam

# execute python script
# export PATH=/uge_mnt/home/toshi/.local//bin:$PATH
# nnictl create --config ./darts_sample.yml
/usr/bin/python darts_sample_exec.py
