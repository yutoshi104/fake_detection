#!/bin/sh
#$ -S /bin/bash # UGE-unique line, sets interpreter shell as bash
#$ -cwd # setting work directory to current working directory, as in, this shell has to be in the same spot as the python script
#$ -jc gs-container_g1_dev # use hss: gtb-container_g1〜g8(A100-640GB)(max24h), gtn-container_g1〜g8(A100-320GB)(max24h), gs-container_g1(DGX2)(max24h), gtn-container_g1_dev〜g4(A100-320GB)(max4h), gs-container_g1_dev(DGX2)(max4h)
# $ -ac d=nvcr-tensorflow-2201-tf2-py3 # setting correct container type  # cat /usr/local/etc/CONTAINER-INFO/tensorflow
# $ -ac d=nvcr-tensorflow-2012-tf2-py3
#$ -ac d=aip-tensorflow-2012-opencv-1

# $ -e /home/toshi/fake_detection/model/OriginalNet_20221121-174136_epoch50/test.sh.e$JOB_ID
# $ -o /home/toshi/fake_detection/model/OriginalNet_20221121-174136_epoch50/test.sh.o$JOB_ID

# load proper environment variables
# . /fefs/opt/dgx/env_set/nvcr-tensorflow-2201-tf2-py3.sh
. /fefs/opt/dgx/env_set/nvcr-tensorflow-2012-tf2-py3.sh

# package install
. ~/proxy.sh
export TF_FORCE_GPU_ALLOW_GROWTH=true

/usr/bin/python -m pip install --upgrade pip
/usr/bin/python -m pip install matplotlib

# execute python script
/usr/bin/python test.py