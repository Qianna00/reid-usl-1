#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29521}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --work-dir /root/vsislab-2/zq/ICCV_hex/spcl/msmt17/try --launcher pytorch ${@:3}
