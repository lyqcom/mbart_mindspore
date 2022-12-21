import torch
import mindspore as M
from mindspore import context
import mindspore.ops as P
from src.mod import MBartConfig, MBartTokenizer, MBartForConditionalGeneration

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

vocab_file = r'/home/mindspore/.cache/huggingface/hub/models--facebook--mbart-large-en-ro/snapshots' \
             r'/2534e987b9ed03c416bbbaefa1a39e3441439bdd/sentencepiece.bpe.model'

bin_file= r'/home/mindspore/.cache/huggingface/hub/models--facebook--mbart-large-en-ro/snapshots' \
          r'/2534e987b9ed03c416bbbaefa1a39e3441439bdd/pytorch_model.bin'

ckpt_file = r'../model.ckpt'

config = MBartConfig(vocab_file=vocab_file)

tok = MBartTokenizer(config)

model = MBartForConditionalGeneration(config)

torch_dict = torch.load(bin_file, torch.device('cpu'))

mindspore_name_2_pytorch = {
    "layer_norm.gamma": "layer_norm.weight",
    "layer_norm.beta": "layer_norm.bias",
    "layernorm_embedding.gamma": "layernorm_embedding.weight",
    "layernorm_embedding.beta": "layernorm_embedding.bias",
    "shared.embedding_table": "shared.weight",
    "lm_head.weight": "model.shared.weight",
    "embed_positions.embedding_table": "embed_positions.weight"
}

mindspore_dict = model.trainable_params()

printer = P.Print()

torch_weights_names = open('torch_ckpt_key.txt', 'w')
mindspore_weights_names = open('mindspore_ckpt_key.txt', 'w')

for name in torch_dict:
    print(name, file=torch_weights_names)

with open('weights_transformation.log', 'w') as f:
    for parm in mindspore_dict:
        parm: M.Parameter
        torch_key = parm.name
        print(torch_key, file=mindspore_weights_names)
        for word in mindspore_name_2_pytorch:
            if word in torch_key:
                torch_key = torch_key.replace(word, mindspore_name_2_pytorch[word])
                break
        print(f'mapping `{torch_key}` from pytorch to `{parm.name}` in mindspore', file=f)
        parm.set_data(M.Tensor(torch_dict[torch_key].numpy()))

M.save_checkpoint(model, ckpt_file)

torch_weights_names.close()
mindspore_weights_names.close()


