"""

example:
    python3 -u preprocess.py \
        --vocab_file ../model.ckpt \
        --result_path ./preprocess_results \
        --src_file ./data/en.txt \
        --tar_file ./data/ro.txt

"""

import argparse
import os
import numpy as np

from src.mod import MBartConfig, MBartTokenizer


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
    parser.add_argument('--result_path', type=str,
                        default='./result')
    parser.add_argument('--src_file', type=str,
                        default='./data/en.txt')
    parser.add_argument('--batch_size', type=int,
                        default=16)
    parser.add_argument('--max_length', type=int,
                        default=256)
    return parser.parse_args()


def generate_bin():
    """
    Generate bin files.
    """

    args = get_args()

    config = MBartConfig(
        vocab_file=args.vocab_file,
        src_lang=args.src_lang,
        tgt_lang=args.tar_lang,
    )

    tokenizer = MBartTokenizer(config)

    dataset = MyDataset(args.src_file, tokenizer, args.batch_size, max_length=args.max_length)

    cur_dir = args.result_path

    input_ids_dir = os.path.join(cur_dir, "input_ids")
    attention_mask_dir = os.path.join(cur_dir, "attention_mask")
    tgt_lang_id = os.path.join(cur_dir, "tgt_lang_id")

    if not os.path.isdir(input_ids_dir):
        os.makedirs(input_ids_dir)
    if not os.path.isdir(attention_mask_dir):
        os.makedirs(attention_mask_dir)
    if not os.path.isdir(tgt_lang_id):
        os.makedirs(tgt_lang_id)

    batch_size = args.batch_size

    for i, (t1, t2) in enumerate(dataset):
        file_name = "transformer_bs_" + str(batch_size) + "_" + str(i) + ".bin"
        t1.asnumpy().tofile(os.path.join(input_ids_dir, file_name))
        t2.asnumpy().tofile(os.path.join(attention_mask_dir, file_name))
        np.array([tokenizer.tgt_lang_id], dtype=np.int).tofile(os.path.join(tgt_lang_id, file_name))


if __name__ == '__main__':
    generate_bin()
