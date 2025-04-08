#!/bin/sh -x

. ./.venv/bin/activate

now=`date "+%F_%T"`
echo $now
mkdir ./log/$now
python ./main_semi.py 2>&1 | tee ./log/$now/log.txt

# move files
if [ -e "loss.png" ]; then
    mv loss.png ./log/$now/
fi

if [ -e "cifar100_pre_train.pth" ]; then
    mv cifar100_pre_train.pth ./models/
fi

deactivate