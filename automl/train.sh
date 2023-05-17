#!/bin/sh
#$ -S /bin/bash # UGE-unique line, sets interpreter shell as bash
#$ -cwd # setting work directory to current working directory, as in, this shell has to be in the same spot as the python script
#$ -jc gs-container_g16_dev # setting smallest possible GPU container for the test job
#$ -ac d=aip-tensorflow-2201-opencv-1

# load proper environment variables
. /fefs/opt/dgx/env_set/nvcr-tensorflow-2201-tf2-py3.sh

# package install
. ~/proxy.sh

/usr/bin/python -m pip install --upgrade pip
/usr/bin/python -m pip install matplotlib
/usr/bin/python -m pip install python-dotenv
/usr/bin/python -m pip install nni pytorch_lightning==2.0.2
/usr/bin/python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 torchmetrics==0.11.4 torchsummary==1.5.1 --extra-index-url https://download.pytorch.org/whl/cu116

/usr/bin/python --version
/usr/bin/python -m pip freeze

# execute python script
/usr/bin/python train.py