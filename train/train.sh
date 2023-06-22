#!/bin/sh
#$ -S /bin/bash # UGE-unique line, sets interpreter shell as bash
#$ -cwd # setting work directory to current working directory, as in, this shell has to be in the same spot as the python script
#$ -jc gtn-container_g8 # setting smallest possible GPU container for the test job
# $ -ac d=nvcr-tensorflow-2201-tf2-py3 # setting correct container type  # cat /usr/local/etc/CONTAINER-INFO/tensorflow
#$ -ac d=aip-tensorflow-2201-opencv-1
# $ -ac d=nvcr-tensorflow-2012-tf2-py3
# $ -ac d=aip-tensorflow-2012-opencv-1

# load proper environment variables
. /fefs/opt/dgx/env_set/nvcr-tensorflow-2201-tf2-py3.sh
# . /fefs/opt/dgx/env_set/nvcr-tensorflow-2012-tf2-py3.sh

# package install
. ~/proxy.sh

/usr/bin/python -m pip install --upgrade pip
# /usr/bin/python -m pip uninstall --yes opencv-python
# /usr/bin/python -m pip install opencv-python
/usr/bin/python -m pip install matplotlib
# /usr/bin/python -m pip install 
# uname –a
# cat /etc/os-release
# cat /etc/centos-release
# cat /etc/redhat-release
# cat /proc/version

pip install python-dotenv

# なんかdockerコンテナを他で変更すると一緒に書き換えられるので、これが必要
# echo y | /usr/bin/python -m pip uninstall tensorflow
# /usr/bin/python -m pip install tensorflow==2.7.0
# echo y | /usr/bin/python -m pip uninstall numpy
# /usr/bin/python -m pip install numpy
# echo y | /usr/bin/python -m pip uninstall keras
# /usr/bin/python -m pip install keras
# echo y | /usr/bin/python -m pip uninstall protobuf
# /usr/bin/python -m pip install protobuf==3.20.1

/usr/bin/python -m pip list

# execute python script
/usr/bin/python train.py