# mbart_mindspore

## 数据下载和预处理

实验使用的数据集为 wmt16-en-ro 的数据集，使用以下命令下载数据。

```bash
git clone https://github.com/rsennrich/wmt16-scripts
cd wmt16-scripts
cd sample
./download_files.sh
./preprocess.sh
```

英文语料为 `./wmt16-scripts/sample/data/newsdev2016.en`

罗马尼亚语语料为 `./wmt16-scripts/sample/data/newsdev2016.ro`

## 训练脚本

调用 train.py 脚本

传入参数:
- is_model_arts: true 或 false，代表是否为 modelarts 环境
- data_dir: 数据所在路径
- vocab_path: 词典所在路径
- result_dir:  保存结果的路径
- device_target: Ascend 或 CPU
- device_id: 单卡运行时的 device id

例如:
```bash
python3 -u train.py \
  --is_model_arts false \
  --data_dir ./data/en_ro \
  --vocab_path ../spm.model \
  --ckpt_path ../model.ckpt \
  --result_dir ./result \
  --device_target Ascend \
  --device_id 0
```

## 推理脚本

调用 eval.py 脚本

传入参数:
- is_model_arts: true 或 false，代表是否为 modelarts 环境
- data_dir: 数据所在路径
- vocab_path: 词典所在路径
- result_dir:  保存结果的路径
- device_target: Ascend 或 CPU
- device_id: 单卡运行时的 device id

例如:
```bash
python3 -u eval.py \
  --is_model_arts false \
  --data_dir ./data/en_ro \
  --vocab_path ../spm.model \
  --ckpt_path ../model.ckpt \
  --result_dir ./result \
  --device_target Ascend \
  --device_id 0
```

