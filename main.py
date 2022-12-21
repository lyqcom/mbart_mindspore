from mindspore import context

from src.models import *
from mindspore.common import set_seed
import mindspore as M
import mindspore.nn as nn

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=2)

set_seed(751)

vocab_file = '../spm.model'
ckpt_file = '../model.ckpt'

config = MBartConfig(
    vocab_file=vocab_file,
    ckpt_file=ckpt_file,
    src_lang='en_XX',
    tgt_lang='ro_RO',
    num_beams=5,
    max_decoder_length=100
)

tokenizer = MBartTokenizer(config)

model: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained(config)

src, tgt = 'like', 'you'

inputs = tokenizer.tokenize(src, tgt)

model.set_train(False)

outputs = model.construct(**inputs)

