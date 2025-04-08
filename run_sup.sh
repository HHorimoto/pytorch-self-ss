#!/bin/sh -x

. ./.venv/bin/activate

now=`date "+%F_%T"`
echo $now
mkdir ./log/$now
python ./main_sup.py 2>&1 | tee ./log/$now/log.txt

# move files
if [ -e "loss.png" ]; then
    mv loss.png ./log/$now/
fi

if [ -e "acc.png" ]; then
    mv acc.png ./log/$now/
fi

deactivate