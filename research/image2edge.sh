#!/bin/sh
#$ -S /bin/bash # UGE-unique line, sets interpreter shell as bash
#$ -cwd # setting work directory to current working directory, as in, this shell has to be in the same spot as the python script
#$ -jc gtn-container_g1_dev # use hss: gtb-container_g1〜g8(A100-640GB)(max24h), gtn-container_g1〜g8(A100-320GB)(max24h), gs-container_g1(DGX2)(max24h), gtn-container_g1_dev〜g4(A100-320GB)(max4h), gs-container_g1_dev(DGX2)(max4h)
# $ -ac d=nvcr-tensorflow-2201-tf2-py3 # setting correct container type  # cat /usr/local/etc/CONTAINER-INFO/tensorflow
#$ -ac d=aip-tensorflow-2201-opencv-1

# load proper environment variables
. /fefs/opt/dgx/env_set/nvcr-tensorflow-2201-tf2-py3.sh

# package install
. ~/proxy.sh

# execute python script
/usr/bin/python image2edge.py
