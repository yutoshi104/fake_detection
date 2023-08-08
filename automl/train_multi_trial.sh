#!/bin/sh
#$ -S /bin/bash # UGE-unique line, sets interpreter shell as bash
#$ -cwd # setting work directory to current working directory, as in, this shell has to be in the same spot as the python script
#$ -jc gtn-container_g1_dev # setting smallest possible GPU container for the test job
#$ -ac d=aip-tensorflow-2012-opencv-1,ep1=5000

# load proper environment variables
. /fefs/opt/dgx/env_set/nvcr-tensorflow-2012-tf2-py3.sh

# package install
. ~/proxy.sh

/usr/bin/python -m pip install --upgrade pip
/usr/bin/python -m pip install matplotlib
/usr/bin/python -m pip install seaborn
/usr/bin/python -m pip install python-dotenv
/usr/bin/python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 torchmetrics==0.11.4 torchsummary==1.5.1 --extra-index-url https://download.pytorch.org/whl/cu116
/usr/bin/python -m pip install nni==2.10 pytorch_lightning==2.0.2
/usr/bin/python -m pip install grad-cam

# echo y | /usr/bin/python -m pip uninstall numpy
# /usr/bin/python -m pip install numpy==1.21
/usr/bin/python --version


# wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
# unzip ngrok-stable-linux-amd64.zip
tar xvzf ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin
ngrok update

# リスニングしているポートの確認
netstat -tuln

ngrok config add-authtoken 2RjTND3hRVxR87c0UEyytS8fPBg_6vEht77Mnzvn3fkyWc1JL
# ngrok start --all --config=./ngrok.yml &

# sleep 5
# curl -s http://localhost:4040/api/tunnels

ngrok http 5000 &
while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' http://localhost:4040/api/tunnels)" != "200" ]]; do sleep 1; done
curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

# execute python script
/usr/bin/python train_multi_trial.py