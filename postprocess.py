"""

example:
    python3 -u preprocess.py \
        --vocab_file ../model.ckpt \
        --result_file ./postprocess_results/out.txt \
        --result_path ./infer_results \
        --result_dir ./data/en.txt \
        --tar_file ./data/ro.txt

"""

import argparse
import os
import numpy as np
from src.mod import MBartConfig, MBartTokenizer
import mindspore as M

class MyDataset:
    """Self Defined dataset."""

    def __init__(self, src_path, tokenizer: MBartTokenizer, batch_size=1, max_length=256):
        lines = list(map(lambda x: x.strip(), open(src_path).readlines()))

        self.inputs = []
        batch = []

        for src_text in lines:
            if src_text != "":
                batch.append(src_text)

                if len(batch) == batch_size:
                    input = tokenizer.tokenize(batch, max_length=max_length)
                    self.inputs.append((input['input_ids'], input['attention_mask']))
                    batch = []

        if batch:
            input = tokenizer.tokenize(batch, max_length=max_length)
            self.inputs.append((input['input_ids'], input['attention_mask']))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


def get_args():
    parser = argparse.ArgumentParser(description='Mbart Mindspore')

    # define 2 parameters for running on modelArts
    # data_url,train_url是固定用于在modelarts上训练的参数，表示数据集的路径和输出模型的路径
    parser.add_argument('--vocab_file', help='model arts or not',
                        default='./data/spm.model')
    parser.add_argument('--src_lang', type=str,
                        default='en_XX')
    parser.add_argument('--tar_lang', type=str,
                        default='ro_RO')
    parser.add_argument('--result_dir', type=str,
                        default='./result')
    parser.add_argument('--result_file', type=str,
                        default='./result')
    parser.add_argument('--tar_file', type=str,
                        default='./data/ro.txt')
    return parser.parse_args()


def generate_output():
    """
    Generate output.
    """
    args = get_args()
    predictions = []
    file_num = len(os.listdir(args.result_dir))

    config = MBartConfig(
        vocab_file=args.vocab_file,
        src_lang=args.src_lang,
        tgt_lang=args.tar_lang,
    )

    tokenizer = MBartTokenizer(config)

    with open(args.result_file, 'w') as f:
        for i in range(file_num):
            batch = "transformer_bs_" + str(args.batch_size) + "_" + str(i) + "_0.bin"
            pred = np.fromfile(os.path.join(args.result_dir, batch), np.int32)
            output = tokenizer.decode(M.Tensor(pred))
            for sentence in output:
                print(sentence, file=f)

    print(f'BLEU score is ', end='')
    os.system(f'sacrebleu {args.result_file} -i {args.tar_file} -m bleu -b -w 4')


if __name__ == "__main__":
    generate_output()
