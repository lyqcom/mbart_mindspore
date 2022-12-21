import argparse

from mindspore import context, export

from src.mod import MBartConfig, MBartTokenizer, MBartForConditionalGeneration, TranslatingCell

import mindspore.numpy as msnp
import mindspore.common.dtype as dtype

def get_args():
    parser = argparse.ArgumentParser(description='Mbart Mindspore')

    # define 2 parameters for running on modelArts
    # data_url,train_url是固定用于在modelarts上训练的参数，表示数据集的路径和输出模型的路径
    parser.add_argument('--device_id', type=int, default=0, help="Device id, default is 0.")
    parser.add_argument('--vocab_path', help='model arts or not',
                        default='./data/spm.model')
    parser.add_argument('--ckpt_file', help='model arts or not',
                        default='./data/model.ckpt')
    parser.add_argument('--device_target', help='device where the code will be implemented (default: Ascend)',
                        type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'], )
    parser.add_argument('--batch_size', type=int, help='batch size',
                        default=1)
    parser.add_argument('--model_dtype', type=str, choices=['float16', 'float32'],
                        default='float16')
    parser.add_argument('--src_lang', type=str,
                        default='en_XX')
    parser.add_argument('--tar_lang', type=str,
                        default='ro_RO')
    parser.add_argument('--max_length', type=int,
                        default=128)
    parser.add_argument('--max_decoder_length', type=int,
                        default=64)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    args.is_model_arts = 'false'
    args.vocab_path = '../spm.model'
    args.ckpt_file = '../model.ckpt'
    args.result_dir = './result'
    args.device_target = 'Ascend'
    args.device_id = 0

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

    vocab_file = args.vocab_path
    ckpt_file = args.ckpt_file

    config = MBartConfig(
        vocab_file=vocab_file,
        ckpt_file=ckpt_file,
        src_lang=args.src_lang,
        tgt_lang=args.tar_lang,
        max_decoder_length=args.max_decoder_length
    )

    tokenizer = MBartTokenizer(config)

    model = MBartForConditionalGeneration(config)

    model.load_weights(config.ckpt_file)
    eval_net = TranslatingCell(model, tokenizer.tgt_lang_id)

    batch_size = args.batch_size
    max_length = args.max_length

    print(eval_net(
        msnp.ones((batch_size, max_length), dtype=dtype.int32),
        msnp.ones((batch_size, max_length), dtype=dtype.int32),
    ))

    # export(eval_net,
    #        msnp.ones((batch_size, max_length), dtype=dtype.int32),
    #        msnp.ones((batch_size, max_length), dtype=dtype.int32),
    #        file_name='mbart-fine-tuned.mindir',
    #        file_format='MINDIR'
    # )
