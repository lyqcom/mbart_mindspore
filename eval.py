import argparse

from mindspore import context, Model

from src.mod import *
from mindspore.common import set_seed
from mindspore.dataset import GeneratorDataset

def moxing_files(src, dst):
    try:
        mox.file.copy_parallel(src, dst)
        print(f"Successfully Download {src} to {dst}")
    except Exception:
        print(f'moxing download {src} to {dst} failed')


def moxing_file(src, dst):
    try:
        mox.file.copy(src, dst)
        print(f"Successfully Download {src} to {dst}")
    except Exception:
        print(f'moxing download {src} to {dst} failed')


class MyDataset:
    """Self Defined dataset."""

    def __init__(self, src_path, tokenizer: MBartTokenizer, batch_size=1):
        lines = list(map(lambda x: x.strip(), open(src_path).readlines()))

        self.inputs = []
        batch = []

        for src_text in lines:
            if src_text != "":
                batch.append(src_text)

                if len(batch) == batch_size:
                    input = tokenizer.tokenize(batch)
                    self.inputs.append((input['input_ids'], input['attention_mask']))
                    batch = []

        if batch:
            input = tokenizer.tokenize(batch)
            self.inputs.append((input['input_ids'], input['attention_mask']))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

def get_args():
    parser = argparse.ArgumentParser(description='Mbart Mindspore')

    # define 2 parameters for running on modelArts
    # data_url,train_url是固定用于在modelarts上训练的参数，表示数据集的路径和输出模型的路径
    parser.add_argument('--device_id', type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--distribute", type=str, default="false", choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument('--is_model_arts', type=str, help='model arts or not', choices=["true", "false"],
                        default='false')
    parser.add_argument('--random_seed', type=int, default=751, help='random seed for mindspore')
    parser.add_argument('--vocab_path', help='model arts or not',
                        default='./data/spm.model')
    parser.add_argument('--ckpt_file', help='model arts or not',
                        default='./data/model.ckpt')
    parser.add_argument('--result_dir', help='model folder to save/load',
                        default='./result')
    parser.add_argument('--test_src', help='path to test.en',
                        default='./data/test.en')
    parser.add_argument('--test_tar', help='path to test.ro',
                        default='./data/test.ro')
    parser.add_argument('--device_target', help='device where the code will be implemented (default: Ascend)',
                        type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'], )
    parser.add_argument('--batch_size', type=int, help='batch size',
                        default=16)
    parser.add_argument('--model_dtype', type=str, choices=['float16', 'float32'],
                        default='float16')
    parser.add_argument('--src_lang', type=str,
                        default='en_XX')
    parser.add_argument('--tar_lang', type=str,
                        default='ro_RO')
    parser.add_argument('--max_decoder_length', type=int,
                        default=64)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    if args.is_model_arts == 'true':
        import moxing as mox

    for dir_name in [args.result_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id, max_call_depth=10000)

    set_seed(args.random_seed)

    vocab_file = args.vocab_path
    ckpt_file = args.ckpt_file
    en_path = args.test_src
    ro_path = args.test_tar
    result_dir = args.result_dir

    out_path = os.path.join(result_dir, 'out.txt')

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
    model = TranslatingCell(model)

    dataset = MyDataset(en_path, tokenizer, batch_size=args.batch_size)

    dataset = GeneratorDataset(dataset, column_names=['input_ids', 'attention_mask'])
    model.set_train(False)
    with open(out_path, 'w') as result_file:
        for batch in dataset.create_tuple_iterator():
            ans = model(*batch, tgt_lang_id=tokenizer.tgt_lang_id)
            output = tokenizer.decode(ans)
            for sentence in output:
                print(sentence)
                print(sentence, file=result_file)

    print(f'BLEU score is ', end='')
    os.system(f'sacrebleu {out_path} -i {ro_path} -m bleu -b -w 4')
    if args.is_model_arts == 'true':
        moxing_files(result_dir, args.result_url)
