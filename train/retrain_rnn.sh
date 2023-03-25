#!/bin/sh
#$ -S /bin/bash # UGE-unique line, sets interpreter shell as bash
#$ -cwd # setting work directory to current working directory, as in, this shell has to be in the same spot as the python script
#$ -jc gs-container_g16 # setting smallest possible GPU container for the test job
# $ -ac d=nvcr-tensorflow-2201-tf2-py3 # setting correct container type  # cat /usr/local/etc/CONTAINER-INFO/tensorflow
# $ -ac d=aip-tensorflow-2201-opencv-1
# $ -ac d=nvcr-tensorflow-2012-tf2-py3
#$ -ac d=aip-tensorflow-2012-opencv-1

# load proper environment variables
# . /fefs/opt/dgx/env_set/nvcr-tensorflow-2201-tf2-py3.sh
. /fefs/opt/dgx/env_set/nvcr-tensorflow-2012-tf2-py3.sh

# package install
. ~/proxy.sh
export TF_FORCE_GPU_ALLOW_GROWTH=true

/usr/bin/python -m pip install --upgrade pip
# /usr/bin/python -m pip uninstall --yes opencv-python
/usr/bin/python -m pip install matplotlib
# /usr/bin/python -m pip install 

pip install python-dotenv

# execute python script
/usr/bin/python retrain_rnn.py
