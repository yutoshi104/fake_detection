#!/bin/sh
#$ -S /bin/bash # UGE-unique line, sets interpreter shell as bash
#$ -cwd # setting work directory to current working directory, as in, this shell has to be in the same spot as the python script
#$ -jc gpu-container_g8.168h # setting smallest possible GPU container for the test job
#$ -ac d=nvcr-tensorflow-2012-tf2-py3

# load proper environment variables
. /fefs/opt/dgx/env_set/nvcr-tensorflow-2012-tf2-py3.sh

# package install
. ~/proxy.sh

/usr/bin/python -m pip install --upgrade pip
pip install pyexiv2

pip install python-dotenv

# execute python script
/usr/bin/python checkExif.py