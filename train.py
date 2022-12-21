import argparse
import os
import time
from typing import Optional

from mindspore import context, Model
import mindspore.communication.management as D

from src.mod import MBartConfig, MBartForConditionalGeneration, MBartTokenizer
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.common import set_seed
import mindspore as M
import mindspore.nn as F
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import ops
from mindspore.communication.management import get_rank, get_group_size
from tqdm import tqdm

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

    def __init__(self, src_path, tgt_path, tokenizer: MBartTokenizer, batch_size=1):
        src_lines = list(map(lambda x: x.strip(), open(src_path).readlines()))
        tgt_lines = list(map(lambda x: x.strip(), open(tgt_path).readlines()))

        self.input_ids = []
        self.labels = []
        self.attention_mask = []
        src_batch = []
        tgt_batch = []

        for idx, (src_text, tgt_text) in enumerate(zip(src_lines, tgt_lines)):
            if src_lines:
                src_batch.append(src_text)
                tgt_batch.append(tgt_text)

                if len(src_batch) == batch_size:
                    inputs = tokenizer.tokenize(src_batch, tgt_batch)
                    self.input_ids.append(inputs['input_ids'])
                    self.attention_mask.append(inputs['attention_mask'])
                    self.labels.append(inputs['labels'])
                    src_batch, tgt_batch = [], []

        if src_batch:
            inputs = tokenizer.tokenize(src_batch, tgt_batch)
            self.input_ids.append(inputs['input_ids'])
            self.attention_mask.append(inputs['attention_mask'])
            self.labels.append(inputs['labels'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = self.labels[idx]
        return input_ids, attention_mask, labels


class NetWithLoss(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(
            self,
            input_ids: M.Tensor = None,  # long
            attention_mask: Optional[M.Tensor] = None,
            labels: Optional[M.Tensor] = None,  # long
    ):
        return self.net(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )[0]


clip_grad = ops.composite.MultitypeFuncGraph("clip_grad")

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 5.


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor]: clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = ops.functional.dtype(grad)
    if clip_type == 0:
        new_grad = ops.composite.clip_by_value(grad,
                                               ops.functional.cast(ops.functional.tuple_to_array((-clip_value,)), dt),
                                               ops.functional.cast(ops.functional.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, ops.functional.cast(ops.functional.tuple_to_array((clip_value,)), dt))
    return new_grad


class MyTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0):
        super(MyTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.hyper_map = ops.composite.HyperMap()

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = ops.functional.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)

        grads = self.hyper_map(ops.functional.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

        grads = self.grad_reducer(grads)
        loss = ops.functional.depend(loss, self.optimizer(grads))
        return loss


def get_args():
    parser = argparse.ArgumentParser(description='Mbart Mindspore')

    # define 2 parameters for running on modelArts
    # data_url,train_url是固定用于在modelarts上训练的参数，表示数据集的路径和输出模型的路径
    parser.add_argument('--device_id', type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--distribute", type=str, default="false", choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument('--is_model_arts', type=str, help='model arts or not', choices=["true", "false"],
                        default='false')
    parser.add_argument('--random_seed', type=int, default=751, help='random seed for mindspore')
    parser.add_argument('--data_url', help='path to training/inference dataset folder',
                        default='./data')
    parser.add_argument('--data_dir', help='path to training/inference dataset folder',
                        default='./data')
    parser.add_argument('--vocab_path', help='model arts or not',
                        default='./data/spm.model')
    parser.add_argument('--ckpt_file', help='model arts or not',
                        default='./data/model.ckpt')
    parser.add_argument('--ckpt_dir', help='model arts or not',
                        default='./data/model.ckpt')
    parser.add_argument('--train_url', help='model folder to save/load',
                        default='./model')
    parser.add_argument('--train_dir', help='model folder to save/load',
                        default='./model')
    parser.add_argument('--result_url', help='model folder to save/load',
                        default='./result')
    parser.add_argument('--result_dir', help='model folder to save/load',
                        default='./result')
    parser.add_argument('--train_src', help='path to train.en',
                        default='./data/train.en')
    parser.add_argument('--train_tar', help='path to train.ro',
                        default='./data/train.ro')
    parser.add_argument('--test_src', help='path to test.en',
                        default='./data/test.en')
    parser.add_argument('--test_tar', help='path to test.ro',
                        default='./data/test.ro')
    parser.add_argument('--device_target', help='device where the code will be implemented (default: Ascend)',
                        type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'], )
    parser.add_argument('--learning_rate', type=float, help='learning rate',
                        default=1e-5)
    parser.add_argument('--num_epochs', type=int, help='number of epochs',
                        default=5)
    parser.add_argument('--batch_size', type=int, help='batch size',
                        default=16)
    parser.add_argument('--model_dtype', type=str, choices=['float16', 'float32'],
                        default='float16')
    parser.add_argument('--src_lang', type=str,
                        default='en_XX')
    parser.add_argument('--tar_lang', type=str,
                        default='ro_RO')
    parser.add_argument('--steps_per_ckpt', type=int,
                        default=20)
    parser.add_argument('--steps_per_callback', type=int,
                        default=1)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    if args.model_dtype == 'float16':
        args.model_dtype = M.common.dtype.float16
    else:
        args.model_dtype = M.common.dtype.float32

    if args.is_model_arts == 'true':
        import moxing as mox

    device_id = int(os.getenv("DEVICE_ID", '0'))
    if args.device_id != device_id:
        device_id = args.device_id
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=device_id)
    if args.distribute == 'true':
        device_num = args.device_num
        rank = device_id % device_num
        print("device_id is {}, rank_id is {}".format(device_id, rank))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        D.init()
    else:
        rank = 0
        device_num = 1

    for dir_name in [args.data_dir, args.result_dir, args.train_dir, args.ckpt_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    set_seed(args.random_seed)

    data_dir = args.data_dir
    if args.is_model_arts == 'true':
        moxing_files(args.data_url, data_dir)

    vocab_file = args.vocab_path
    ckpt_file = args.ckpt_file
    en_path = args.train_src
    ro_path = args.train_tar
    result_dir = args.result_dir

    config = MBartConfig(
        vocab_file=vocab_file,
        ckpt_file=ckpt_file,
        src_lang=args.src_lang,
        tgt_lang=args.tar_lang,
        model_dtype=args.model_dtype
    )

    tokenizer = MBartTokenizer(config)

    model = MBartForConditionalGeneration(config)

    model.load_weights(config.ckpt_file)

    loss_net = NetWithLoss(model)
    opt = F.Adam(model.trainable_params(), learning_rate=args.learning_rate)

    if args.distribute == 'true':
        dataset = MyDataset(en_path, ro_path, tokenizer, args.batch_size)
        dataset = ds.GeneratorDataset(dataset,
                                      column_names=["input_ids", "attention_mask", "labels"],
                                      shard_id=get_rank(),
                                      num_shards=get_group_size()
                                      )
    else:
        dataset = MyDataset(en_path, ro_path, tokenizer, args.batch_size)
        dataset = ds.GeneratorDataset(dataset,
                                      column_names=["input_ids", "attention_mask", "labels"],
                                      )

    num_epochs = args.num_epochs

    # steps_per_callback = args.steps_per_callback
    # callback = [TimeMonitor(steps_per_callback), LossMonitor(steps_per_callback)]
    #
    # config_ck = CheckpointConfig(save_checkpoint_steps=args.steps_per_ckpt, keep_checkpoint_max=3)
    # ckpt_cb = ModelCheckpoint(prefix="MBart", config=config_ck)
    # callback.append(ckpt_cb)
    #
    # model = Model(loss_net, optimizer=opt, amp_level="O2")
    # model.train(num_epochs, dataset, callbacks=callback, dataset_sink_mode=False)

    train_one_step = F.TrainOneStepWithLossScaleCell(loss_net, opt, scale_sense=M.Tensor(1024, dtype=M.common.dtype.float16))
    tot_steps = dataset.get_dataset_size()

    for i in range(args.num_epochs):
        for idx, data in enumerate(dataset):
            tic = time.time()
            loss = train_one_step(*data)
            toc = time.time()
            print(f'epoch: {i+1}/{args.num_epochs}, step: {idx+1}/{tot_steps}, time: {toc-tic:.2f}s, loss: {loss}')
        new_ckpt_path = os.path.join(args.ckpt_dir, f'model_ckpt_epoch_{i+1}.ckpt')
        M.save_checkpoint(model, new_ckpt_path)

    if args.is_model_arts == 'true':
        moxing_files(result_dir, args.result_url)
