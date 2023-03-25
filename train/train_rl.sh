#!/bin/sh
#$ -S /bin/bash # UGE-unique line, sets interpreter shell as bash
#$ -cwd # setting work directory to current working directory, as in, this shell has to be in the same spot as the python script
#$ -jc gs-container_g16_dev # setting smallest possible GPU container for the test job
#$ -ac d=nvcr-tensorflow-2207-tf2-py3 # setting correct container type  # cat /usr/local/etc/CONTAINER-INFO/tensorflow

# load proper environment variables
. /fefs/opt/dgx/env_set/nvcr-tensorflow-2207-tf2-py3.sh

# package install
. ~/proxy.sh

# /usr/bin/python -m pip install --upgrade pip
# /usr/bin/python -m pip install matplotlib
# /usr/bin/python -m pip install autokeras

pip list
pip install matplotlib
pip install autokeras

# execute python script
/usr/bin/python train_rl.py