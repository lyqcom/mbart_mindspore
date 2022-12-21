#!/bin/bash

# example: ./run_infer310.sh ../mindir/Mbart_graph.mindir ../data/en.txt ../data/ro.txt

# preprocess

python3 -u preprocess.py \
        --vocab_file ../model.ckpt \
        --result_path ./preprocess_results \
        --src_file "$2" \
        --tar_file "$3" &> preprocess.log

# build 310 exec

cd ../inference_310 || exit
bash build.sh &> build.log

# run 310 exec

cd - || exit
if [ -d result_Files ]; then
    rm -rf ./result_Files
fi

if [ -d time_Result ]; then
    rm -rf ./time_Result
fi

mkdir result_Files
mkdir time_Result

../ascend310_infer/out/main --mindir_path=$1 --input0_path=./preprocess_Result/input_ids --input1_path=./preprocess_Result/attention_mask &> infer.log

# postprocess

python3 -u preprocess.py \
        --vocab_file ../model.ckpt \
        --result_file ./postprocess_results/out.txt \
        --result_dir ./result_Files/ \
        --tar_file $3 &> postprocess.log
