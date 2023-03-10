#!/bin/bash
# ./run_distribute_train.sh ./data 2

DATA_PATH=$1
export DATA_PATH=${DATA_PATH}
RANK_SIZE=$2

EXEC_PATH=$(pwd)

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_${RANK_SIZE}pcs

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir ./device$i
    cp ./resnet50_distributed_training.py ./resnet.py ./device$i
    cd ./device$i || exit
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    pytest -s -v ./resnet50_distributed_training.py > train.log$i 2>&1 &
    cd ../
done
rm -rf device0
mkdir device0
cp ./resnet50_distributed_training.py ./resnet.py ./device0
cd ./device0 || exit
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
pytest -s -v ./resnet50_distributed_training.py > train.log0 2>&1
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi

cd ../