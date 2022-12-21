#!/bin/bash

source /root/env.sh

export GLOG_v=1

time=`date +%Y-%m-%d_%H-%M-%S`
log_name=./log/eval/log_$time.log
err_name=./log/eval/log_$time.err

echo log to $log_name
echo err to $err_name

python3 -u ../eval.py \
    --distribute false \
    --is_model_arts false \
    --vocab_path '../../spm.model' \
    --ckpt_file '../ckpt/model_ckpt_epoch_10.ckpt' \
    --test_src '../data/test.en' \
    --test_tar '../data/test.ro' \
    --result_dir './result' \
    --device_target 'CPU' \
    --random_seed 751 \
    --src_lang en_XX \
    --tar_lang ro_RO \
    --batch_size 1 \
    --max_decoder_length 40 \
    --device_id 0 1>"$log_name" 2>"$err_name" &


# VIP_LOG ./log/eval/log_2022-12-10_18-24-55.log

# log to ./log/train/log_2022-12-11_13-29-14.log
# err to ./log/train/log_2022-12-11_13-29-14.err

# log to ./log/train/log_2022-12-11_14-14-23.log
# err to ./log/train/log_2022-12-11_14-14-23.err

# log to ./log/eval/log_2022-12-11_14-34-18.log
# err to ./log/eval/log_2022-12-11_14-34-18.err

# log to ./log/eval/log_2022-12-11_14-43-09.log
# err to ./log/eval/log_2022-12-11_14-43-09.err

# debug
# log to ./log/eval/log_2022-12-11_14-54-39.log
# err to ./log/eval/log_2022-12-11_14-54-39.err

# log to ./log/eval/log_2022-12-11_14-46-43.log
# err to ./log/eval/log_2022-12-11_14-46-43.err

# log to ./log/eval/log_2022-12-11_15-54-12.log
# err to ./log/eval/log_2022-12-11_15-54-12.err

# log to ./log/eval/log_2022-12-11_16-04-48.log
# err to ./log/eval/log_2022-12-11_16-04-48.err

