#!/bin/bash

source /root/env.sh

export GLOG_v=1

time=`date +%Y-%m-%d_%H-%M-%S`
log_name=./log/train/log_$time.log
err_name=./log/train/log_$time.err

echo log to $log_name
echo err to $err_name

python3 -u ../train.py \
    --distribute false \
    --is_model_arts false \
    --data_dir ./data \
    --ckpt_dir ../ckpt \
    --vocab_path '../../spm.model' \
    --ckpt_file '../../model.ckpt' \
    --train_src '../data/en.txt' \
    --train_tar '../data/ro.txt' \
    --result_dir './result' \
    --device_target 'Ascend' \
    --random_seed 751 \
    --src_lang en_XX \
    --src_lang ro_RO \
    --model_dtype float32 \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --num_epoch 10 \
    --device_id 0 1>"$log_name" 2>"$err_name" &

# ./log/train/log_2022-12-11_10-15-39.log
# ./log/train/log_2022-12-11_10-15-39.err

# log to ./log/train/log_2022-12-11_11-35-25.log
# err to ./log/train/log_2022-12-11_11-35-25.err

# log to ./log/train/log_2022-12-11_13-10-37.log
# err to ./log/train/log_2022-12-11_13-10-37.err

# log to ./log/train/log_2022-12-11_13-16-59.log
# err to ./log/train/log_2022-12-11_13-16-59.err

# log to ./log/train/log_2022-12-11_14-13-56.log
# err to ./log/train/log_2022-12-11_14-13-56.err

# ./log/train/log_2022-12-11_10-18-33.log
# ./log/train/log_2022-12-11_10-18-33.err

# log to ./log/train/log_2022-12-11_14-36-29.log
# err to ./log/train/log_2022-12-11_14-36-29.err

# log to ./log/train/log_2022-12-11_15-07-04.log
# err to ./log/train/log_2022-12-11_15-07-04.err

# log to ./log/train/log_2022-12-11_15-13-16.log
# err to ./log/train/log_2022-12-11_15-13-16.err



