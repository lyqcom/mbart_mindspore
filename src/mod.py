import math
import os

import numpy as np
import time
import mindspore as M
import mindspore.nn as F
import mindspore.ops as P
import mindspore.numpy as msnp
import mindspore.common.dtype as dtype
from typing import Optional, Tuple, Union, Dict, List
import subprocess
import sys
import mindspore.ops.functional as FN

from mindspore import ms_function, ms_class

FAIRSEQ_LANGUAGE_CODES = [
    "ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX",
    "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN",
    "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT",
    "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO",
    "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"
]


class SoftmaxCrossEntropyExpand(F.Cell):
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = P.Exp()
        self.sum = P.ReduceSum(keep_dims=True)
        self.onehot = P.OneHot()
        self.on_value = M.Tensor(1.0, dtype.float32)
        self.off_value = M.Tensor(0.0, dtype.float32)
        self.div = P.RealDiv()
        self.log = P.Log()
        self.sum_cross_entropy = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.mean = P.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = P.ReduceMax(keep_dims=True)
        self.sub = P.Sub()

    def construct(self, logit, label):
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, P.shape(logit)[1], self.on_value, self.off_value)
        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(P.scalar_to_array(-1.0), loss)
        loss = self.mean(loss, -1)
        return loss


@ms_class
class MBartConfig:
    r"""
    This is the configuration class to store the configuration of a [`MBartModel`]. It is used to instantiate an MBART
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MBART
    [facebook/mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the MBART model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MBartModel`] or [`TFMBartModel`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop: (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models)
        forced_eos_token_id (`int`, *optional*, defaults to 2):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`.
    """

    model_type = "mbart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
            self,
            *,
            vocab_file=None,
            ckpt_dir=None,
            ckpt_file=None,
            src_lang='en_XX',
            tgt_lang='ro_RO',
            vocab_size=250027,
            max_decoder_length=1024,
            max_position_embeddings=1024,
            encoder_layers=12,
            encoder_ffn_dim=4096,
            encoder_attention_heads=16,
            decoder_layers=12,
            decoder_ffn_dim=4096,
            decoder_attention_heads=16,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            use_cache=True,
            is_encoder_decoder=True,
            activation_function="gelu",
            d_model=1024,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            init_std=0.02,
            classifier_dropout=0.0,
            scale_embedding=True,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            forced_eos_token_id=2,
            output_attentions=False,
            output_hidden_states=False,
            add_bias_logits=False,
            add_cross_attention=False,
            add_final_layer_norm=True,
            architectures=None,
            bad_words_ids=None,
            chunk_size_feed_forward=0,
            classif_dropout=0.1,
            decoder_start_token_id=250020,
            extra_pos_embeddings=2,
            finetuning_task=None,
            save_ckpt=False,
            normalize_before=True,
            normalize_embedding=True,
            num_beam_groups=1,
            num_beams=5,
            num_hidden_layers=12,
            output_past=True,
            repetition_penalty=1.0,
            sep_token_id=None,
            tie_words_embeddings=True,
            top_k=50,
            top_p=1.,
            length_penalty=1.0,
            model_dtype=dtype.float16,
    ):
        self.tgt_lang = tgt_lang
        self.src_lang = src_lang
        self.max_decoder_length = max_decoder_length
        self.model_dtype = model_dtype
        self.ckpt_file = ckpt_file
        self.ckpt_dir = ckpt_dir
        self.length_penalty = length_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.tie_words_embeddings = tie_words_embeddings
        self.sep_token_id = sep_token_id
        self.repetition_penalty = repetition_penalty
        self.output_past = output_past
        self.num_hidden_layers = num_hidden_layers
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.normalize_embedding = normalize_embedding
        self.normalize_before = normalize_before
        self.save_ckpt = save_ckpt
        self.finetuning_task = finetuning_task
        self.extra_pos_embeddings = extra_pos_embeddings
        self.decoder_start_token_id = decoder_start_token_id
        self.classif_dropout = classif_dropout
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.bad_words_ids = bad_words_ids
        self.architectures = architectures
        self.add_final_layer_norm = add_final_layer_norm
        self.add_cross_attention = add_cross_attention
        self.add_bias_logits = add_bias_logits
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.forced_eos_token_id = forced_eos_token_id
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states


@ms_function
def _expand_mask(mask: M.Tensor, ftype: M.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    broadcaster = P.BroadcastTo((bsz, 1, tgt_len, src_len))

    expanded_mask = broadcaster(mask.expand_dims(axis=1).expand_dims(axis=2)).astype(ftype)

    inverted_mask = 1.0 - expanded_mask
    if ftype == dtype.float16:
        val = -65500.0
    else:
        val = -1e9

    return FN.select(inverted_mask.astype(dtype.bool_),
                     msnp.full(inverted_mask.shape, val, dtype=ftype),
                     inverted_mask.astype(ftype)
                     )


# Copied from transformers.models.bart.modeling_bart._expand_mask
@ms_function
def _make_causal_mask(input_ids_shape: Tuple, ftype: dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bidirectional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    if ftype == dtype.float16:
        val = -65500.0
    else:
        val = -1e9
    mask = msnp.full((tgt_len, tgt_len), val, ftype)
    mask_cond = msnp.arange(mask.shape[-1])
    mask = FN.select((mask_cond < (mask_cond + 1).view(mask.shape[-1], 1)),
                     msnp.full(mask.shape, 0, dtype=ftype),
                     mask
                     )
    mask = mask.masked_fill(mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), 0)
    mask = mask.astype(ftype)

    if past_key_values_length > 0:
        mask = P.Concat(axis=-1)((msnp.zeros((tgt_len, past_key_values_length), dtype=ftype), mask))
    cast_op = P.BroadcastTo((bsz, 1, tgt_len, tgt_len + past_key_values_length))
    return cast_op(mask[None, None, :, :])


@ms_function
def shift_tokens_right(a: M.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    # unsqueeze_op = P.ExpandDims()
    # concat_op = P.Concat(axis=-1)
    #
    # c = a[:, :-1]
    # d = unsqueeze_op(a[:, -1], -1)

    # return concat_op((c, d))
    return a


class BasePreTrainedModel(F.Cell):
    def __init__(self, config: MBartConfig):
        super(BasePreTrainedModel, self).__init__()
        self.config = config

    @classmethod
    def from_pretrained(cls, config: MBartConfig):
        model = cls(config)
        failed = model.load_weights(config.ckpt_file)
        return model

    def load_weights(self, ckpt_file):
        self.to_float(self.config.model_dtype)
        failed = M.load_param_into_net(self, M.load_checkpoint(ckpt_file_name=ckpt_file))
        return failed

    def save_weights(self, ckpt_file: str = None):
        ckpt_dir = self.config.ckpt_dir if self.config.ckpt_dir else './'
        if ckpt_file is None:
            ckpt_file = f'checkpoint_{time.strftime("%Y_%m_%d_%H_%M", time.localtime())}.ckpt'
        M.save_checkpoint(self, os.path.join(ckpt_dir, ckpt_file))


class MBartLearnedPositionalEmbedding(F.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_size: int):
        # MBart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super(MBartLearnedPositionalEmbedding, self).__init__(num_embeddings + self.offset, embedding_size)

    def construct(self, input_ids_shape: Tuple, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = msnp.arange(past_key_values_length, past_key_values_length + seq_len)
        return super(MBartLearnedPositionalEmbedding, self).construct(positions + self.offset)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->MBart
class MBartAttention(F.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            has_bias: bool = True,
    ):
        super(MBartAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = F.Dense(embed_dim, embed_dim, has_bias=has_bias)
        self.v_proj = F.Dense(embed_dim, embed_dim, has_bias=has_bias)
        self.q_proj = F.Dense(embed_dim, embed_dim, has_bias=has_bias)
        self.out_proj = F.Dense(embed_dim, embed_dim, has_bias=has_bias)

        self.dropout_layer = F.Dropout(keep_prob=1 - self.dropout)

    def _shape(self, tensor: M.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        # mindspore 没有 torch.Tensor.contiguous() 对应的算子
        # tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def construct(
            self,
            hidden_states: M.Tensor,
            key_value_states: Optional[M.Tensor] = None,
            past_key_value: Optional[Tuple[M.Tensor]] = None,
            attention_mask: Optional[M.Tensor] = None,
            layer_head_mask: Optional[M.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[M.Tensor, Optional[M.Tensor], Optional[Tuple[M.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = P.Concat(axis=2)((past_key_value[0], key_states))
            value_states = P.Concat(axis=2)((past_key_value[1], value_states))
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = P.BatchMatMul()(query_states, FN.transpose(key_states, (0, 2, 1)))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.Softmax(axis=-1)(attn_weights)

        if layer_head_mask is not None:
            if layer_head_mask.shape != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.shape}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = self.dropout_layer(attn_weights)

        attn_output = P.BatchMatMul()(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output`
        # can be partitioned across GPUs when using tensor-parallelism.
        attn_output = FN.reshape(attn_output, (bsz, tgt_len, self.embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class MBartEncoderLayer(F.Cell):
    def __init__(self, config: MBartConfig):
        super(MBartEncoderLayer, self).__init__()
        self.config = config
        self.embed_dim = config.d_model
        self.self_attn = MBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = F.LayerNorm((self.embed_dim,), epsilon=1e-5)
        self.dropout = config.dropout
        self.activation_fn = F.GELU()
        self.activation_dropout = config.activation_dropout
        self.fc1 = F.Dense(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = F.Dense(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = F.LayerNorm((self.embed_dim,), epsilon=1e-5)

        self.dropout_layer = F.Dropout(keep_prob=1 - self.dropout)
        self.activation_dropout_layer = F.Dropout(keep_prob=1 - self.activation_dropout)
        self.isinf = P.IsInf()

    def construct(
            self,
            hidden_states: M.Tensor,
            attention_mask: M.Tensor,
            layer_head_mask: M.Tensor,
            output_attentions: bool = False,
    ) -> Tuple:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape *(seq_len, batch, embed_dim)*
            attention_mask (`torch.FloatTensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                *(encoder_attention_heads,)*.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states

        ftype = hidden_states.dtype

        if ftype == dtype.float16 and (
                self.isinf(hidden_states).any() or P.isnan(hidden_states).any()
        ):
            if ftype == dtype.float16:
                val = 65500.0
            else:
                val = 1e9
            clamp_value = msnp.array(val - 1000, dtype=ftype)
            # clamp_value = M.Tensor(elem, dtype=ftype)
            hidden_states = P.clip_by_value(
                hidden_states,
                clip_value_min=-clamp_value,
                clip_value_max=clamp_value
            )

        return (hidden_states, attn_weights) if output_attentions else (hidden_states,)


class MBartDecoderLayer(F.Cell):
    def __init__(self, config: MBartConfig):
        super(MBartDecoderLayer, self).__init__()
        self.config = config
        self.embed_dim = config.d_model

        self.self_attn = MBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = F.GELU()
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = F.LayerNorm((self.embed_dim,), epsilon=1e-5)
        self.encoder_attn = MBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = F.LayerNorm((self.embed_dim,), epsilon=1e-5)
        self.fc1 = F.Dense(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = F.Dense(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = F.LayerNorm((self.embed_dim,), epsilon=1e-5)

        self.dropout_layer = F.Dropout(keep_prob=1 - self.dropout)
        self.activation_dropout_layer = F.Dropout(keep_prob=1 - self.activation_dropout)

    def construct(
            self,
            hidden_states: M.Tensor,
            attention_mask: Optional[M.Tensor] = None,
            encoder_hidden_states: Optional[M.Tensor] = None,
            encoder_attention_mask: Optional[M.Tensor] = None,
            layer_head_mask: Optional[M.Tensor] = None,
            cross_attn_layer_head_mask: Optional[M.Tensor] = None,
            past_key_value: Optional[Tuple[M.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ) -> Tuple:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape *(seq_len, batch, embed_dim)*
            attention_mask (`torch.FloatTensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape *(seq_len, batch, embed_dim)*
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                *(encoder_attention_heads,)*.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size *(decoder_attention_heads,)*.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`Optional[bool]`)
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = self.dropout_layer(hidden_states)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MBartEncoder(BasePreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`MBartEncoderLayer`].

    Args:
        config: MBartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self,
                 config: MBartConfig,
                 embed_tokens: Optional[F.Embedding] = None,
                 embed_positions: Optional[F.Embedding] = None
                 ):
        super(MBartEncoder, self).__init__(config)
        self.config = config
        self.dropout = config.dropout
        self.layer_dropout = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = F.Embedding(config.vocab_size, embed_dim, padding_idx=self.padding_idx)

        if embed_positions is not None:
            self.embed_positions = embed_positions
        else:
            self.embed_positions = MBartLearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
            )
        self.layers = F.CellList([MBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = F.LayerNorm((embed_dim,), epsilon=1e-5)
        self.layer_norm = F.LayerNorm((config.d_model,), epsilon=1e-5)

        self.dropout_layer = F.Dropout(keep_prob=1 - self.dropout)

        self.output_hidden_states = self.config.output_hidden_states
        self.output_attentions = self.config.output_attentions
        self.print = P.Print()

    def construct(
            self,
            input_ids: M.Tensor = None,  # dtype = long
            attention_mask: Optional[M.Tensor] = None,
            head_mask: Optional[M.Tensor] = None,
            inputs_embeds: Optional[M.Tensor] = None,  # dtype = float32
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> Dict:  # Tuple
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`MBartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
        """
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout_layer(hidden_states)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != len(self.layers):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.shape[0]}."
                )
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # dropout_probability = random.uniform(0, 1)
            # if self.training and (dropout_probability < self.layerdrop):  # skip the layer
            #     layer_outputs = (None, None)
            # else:
            #     if self.gradient_checkpointing and self.training:
            #
            #         def create_custom_forward(module):
            #             def custom_forward(*inputs):
            #                 return module(*inputs, output_attentions)
            #
            #             return custom_forward
            #
            #         layer_outputs = torch.utils.checkpoint.checkpoint(
            #             create_custom_forward(encoder_layer),
            #             hidden_states,
            #             attention_mask,
            #             (head_mask[idx] if head_mask is not None else None),
            #         )
            #     else:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        ret = {
            'last_hidden_state': hidden_states,
            'hidden_states': encoder_states,
            'attentions': all_attentions
        }
        return ret


class MBartDecoder(BasePreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MBartDecoderLayer`]

    Args:
        config: MBartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: MBartConfig,
                 embed_tokens: Optional[F.Embedding] = None,
                 embed_positions: Optional[MBartLearnedPositionalEmbedding] = None
                 ):
        super(MBartDecoder, self).__init__(config)
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = F.Embedding(config.vocab_size, config.d_model, padding_idx=self.padding_idx)

        if embed_positions is not None:
            self.embed_positions = embed_positions
        else:
            self.embed_positions = MBartLearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
            )

        self.layers = F.CellList([MBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = F.LayerNorm((config.d_model,), epsilon=1e-5)
        self.layer_norm = F.LayerNorm((config.d_model,), epsilon=1e-5)

        self.dropout_layer = F.Dropout(keep_prob=1 - self.dropout)

        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.use_cache = self.config.use_cache

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def construct(
            self,
            input_ids: M.Tensor = None,  # long
            attention_mask: Optional[M.Tensor] = None,
            encoder_hidden_states: Optional[M.Tensor] = None,  # float32
            encoder_attention_mask: Optional[M.Tensor] = None,  # long
            head_mask: Optional[M.Tensor] = None,
            cross_attn_head_mask: Optional[M.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[M.Tensor]]] = None,  # float32
            inputs_embeds: Optional[M.Tensor] = None,  # float32
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Dict:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`MBartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.use_cache

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = self.dropout_layer(hidden_states)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.shape[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.shape[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # dropout_probability = random.uniform(0, 1)
            # if self.training and (dropout_probability < self.layerdrop):
            #     continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        return hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions
        # return {
        #     'last_hidden_state': hidden_states,
        #     'past_key_values': next_cache,
        #     'hidden_states': all_hidden_states,
        #     'attentions': all_self_attns,
        #     'cross_attentions': all_cross_attentions
        # }


class MBartModel(BasePreTrainedModel):
    def __init__(self, config: MBartConfig):
        super(MBartModel, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = F.Embedding(vocab_size, config.d_model, padding_idx=padding_idx)

        self.encoder = MBartEncoder(config, self.shared)
        self.decoder = MBartDecoder(config, self.shared)

        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.use_cache = self.config.use_cache
        self.pad_token_id = self.config.pad_token_id

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def construct(
            self,
            input_ids: M.Tensor = None,  # long
            attention_mask: Optional[M.Tensor] = None,
            decoder_input_ids: Optional[M.Tensor] = None,  # long
            decoder_attention_mask: Optional[M.Tensor] = None,  # long
            head_mask: Optional[M.Tensor] = None,
            decoder_head_mask: Optional[M.Tensor] = None,
            cross_attn_head_mask: Optional[M.Tensor] = None,
            encoder_outputs: Dict = None,  # float32
            past_key_values: Union[Tuple[M.Tensor], Dict] = None,  # float32
            inputs_embeds: Optional[M.Tensor] = None,  # float32
            decoder_inputs_embeds: Optional[M.Tensor] = None,  # float32
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> Dict:
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.use_cache

        # different to other models, MBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.pad_token_id)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs['last_hidden_state']
            if isinstance(encoder_outputs, dict) else encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return decoder_outputs[0], decoder_outputs[1], decoder_outputs[2], \
               decoder_outputs[3], decoder_outputs[4], encoder_outputs['last_hidden_state'], \
               encoder_outputs['hidden_states'], encoder_outputs['attentions']
        # return {
        #     'last_hidden_state': decoder_outputs['last_hidden_state'],
        #     'past_key_values': decoder_outputs['past_key_values'],
        #     'decoder_hidden_states': decoder_outputs['hidden_states'],
        #     'decoder_attentions': decoder_outputs['attentions'],
        #     'cross_attentions': decoder_outputs['cross_attentions'],
        #     'encoder_last_hidden_state': encoder_outputs['last_hidden_state'],
        #     'encoder_hidden_states': encoder_outputs['hidden_states'],
        #     'encoder_attentions': encoder_outputs['attentions']
        # }


class MBartForConditionalGeneration(BasePreTrainedModel):

    def __init__(self, config: MBartConfig):
        super(MBartForConditionalGeneration, self).__init__(config)
        self.model = MBartModel(config)
        self.final_logits_bias = msnp.zeros((1, self.model.shared.vocab_size), dtype=config.model_dtype)
        self.lm_head = F.Dense(config.d_model, self.model.shared.vocab_size, has_bias=False)
        self.vocab_size = self.config.vocab_size
        self.pad_token_id = self.config.pad_token_id
        self.loss_fct = F.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.onehot = P.OneHot()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def construct(
            self,
            input_ids: M.Tensor = None,  # long
            attention_mask: Optional[M.Tensor] = None,
            decoder_input_ids: Optional[M.Tensor] = None,  # long
            decoder_attention_mask: Optional[M.Tensor] = None,  # long
            head_mask: Optional[M.Tensor] = None,
            decoder_head_mask: Optional[M.Tensor] = None,
            cross_attn_head_mask: Optional[M.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[M.Tensor]]] = None,  # float32
            past: Optional[Tuple[Tuple[M.Tensor]]] = None,  # float32
            inputs_embeds: Optional[M.Tensor] = None,  # float32
            decoder_inputs_embeds: Optional[M.Tensor] = None,  # float32
            labels: Optional[M.Tensor] = None,  # long
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            last_output=None
    ) -> Union[Dict, List]:  # float32
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.pad_token_id)

        if last_output is not None:
            past = last_output[2]

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.vocab_size), labels.view(-1))

        return masked_lm_loss, lm_logits, outputs[1], outputs[2], outputs[3], \
               outputs[4], outputs[5], outputs[6], outputs[7]
        #        {
        #     'loss': masked_lm_loss,
        #     'logits': lm_logits,
        #     'past_key_values': outputs['past_key_values'],
        #     'decoder_hidden_states': outputs['decoder_hidden_states'],
        #     'decoder_attentions': outputs['decoder_attentions'],
        #     'cross_attentions': outputs['cross_attentions'],
        #     'encoder_last_hidden_state': outputs['encoder_last_hidden_state'],
        #     'encoder_hidden_states': outputs['encoder_hidden_states'],
        #     'encoder_attentions': outputs['encoder_attentions']
        # }

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: M.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class TranslatingCell(F.Cell):
    def __init__(self, model, tgt_lang_id):
        super(TranslatingCell, self).__init__()
        self.model: MBartForConditionalGeneration = model
        self.encoder = self.model.get_encoder()
        self.max_length = model.config.max_decoder_length
        self.eos_token = model.config.eos_token_id
        self.tgt_lang_id = tgt_lang_id

    def construct(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        batch_size = input_ids.shape[0]
        input_ids = msnp.full((batch_size, 1), self.tgt_lang_id, dtype=msnp.dtypes.int32)
        expanded_return_idx = msnp.arange(batch_size)
        attention_mask = attention_mask[expanded_return_idx]
        input_ids = input_ids[expanded_return_idx]
        encoder_outputs['last_hidden_state'] = encoder_outputs['last_hidden_state'][expanded_return_idx]

        done = msnp.full((batch_size,), False, dtype=dtype.bool_)
        last_output = None

        for cur_len in range(1, self.max_length + 1):
            if last_output is not None:
                input_ids = input_ids[:, -1:]
            # return {
            #     "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            #     "encoder_outputs": encoder_outputs,
            #     "past_key_values": past,
            #     "decoder_input_ids": decoder_input_ids,
            #     "attention_mask": attention_mask,
            #     "head_mask": head_mask,
            #     "decoder_head_mask": decoder_head_mask,
            #     "cross_attn_head_mask": cross_attn_head_mask,
            #     "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            # }
            # model_inputs = self.model.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, encoder_outputs=encoder_outputs)
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=input_ids,
                attention_mask=attention_mask,
                last_output=last_output
            )
            last_output=outputs
            # 取最后一个timestep的输出 (batch_size*num_beams, vocab_size)
            next_token_logits = outputs[1][:, -1:]
            scores = P.LogSoftmax(axis=-1)(next_token_logits)
            # 转成(batch_size, num_beams * vocab_size)
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, 2 * num_beams)
            scores = scores.view(batch_size, -1)
            _, next_tokens = P.TopK(sorted=True)(scores, 1)
            input_ids = P.concat((input_ids, next_tokens), axis=-1)
            done = FN.logical_or(done, (next_tokens.view(-1) == self.eos_token))
            if done.all():
                break

        return input_ids


class MBartTokenizer:
    model_input_names = ["input_ids", "attention_mask"]
    prefix_tokens = []
    suffix_tokens = []

    def __init__(
            self,
            config: MBartConfig,
            *,
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
    ):

        # Mask token behave like a normal word, i.e. include the space before it
        try:
            from tokenizers import AddedToken
        except ImportError:
            print('installing tokenizers')
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', 'tokenizers'])
            from tokenizers import AddedToken

        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.vocab_file = config.vocab_file

        try:
            import sentencepiece as spm
        except ImportError:
            print('installing spm')
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', 'sentencepiece'])
            import sentencepiece as spm

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

        # Original fairseq vocab and spm vocab must be "aligned":
        # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
        # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
        # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
        # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

        # Mimic fairseq token-to-id alignment for the first 4 token
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.fairseq_offset = 1

        self.sp_model_size = len(self.sp_model)
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset

        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
        self._additional_special_tokens = list(self.lang_code_to_id.keys())

        self.cur_lang_code = None
        self._src_lang = config.src_lang
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.tgt_lang = config.tgt_lang
        self.src_lang = config.src_lang
        config.vocab_size = self.vocab_size

    def convert_tokens_to_ids(self, tokens: Union[str, List[str], List[List[str]]]) -> Union[
        int, List[int], List[List[int]]]:
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)

        if isinstance(tokens[0], str):
            return [self._convert_token_to_id(token) for token in tokens]

        if isinstance(tokens[0][0], str):
            return [self.convert_tokens_to_ids(tokens) for tokens in tokens]

        raise TypeError("expected `str`, `List[str]`, `List[List[str]]` for the input")

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, tok_id):
        """Converts a tok_id (int) in a token using the vocab."""
        if tok_id in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[tok_id]
        return self.sp_model.IdToPiece(tok_id - self.fairseq_offset)

    def convert_a_sentence_of_ids_to_tokens(self, tok_ids) -> List[str]:
        ret = []
        for tok_id in tok_ids:
            if tok_id == self.eos_token_id:
                break
            else:
                ret.append(self._convert_id_to_token(int(tok_id)))
        return ret

    def convert_ids_to_tokens(self, input_ids: M.Tensor) -> Union[List[List[str]], List[str]]:
        if input_ids.dim() == 2:
            return [self.convert_a_sentence_of_ids_to_tokens(sentence) for sentence in input_ids]
        if input_ids.dim() == 1:
            return self.convert_a_sentence_of_ids_to_tokens(input_ids)
        raise ValueError(f"input_ids's dim must be 1 or 2, but given {input_ids.dim()}")

    def tokenize(self, text: Union[str, List[str]], ref: Union[str, List[str]] = None, max_length=128) -> Dict:
        if isinstance(text, str):
            text = [text]
        if isinstance(ref, str):
            ref = [ref]
        batch_size = len(text)
        tokens = self.sp_model.Encode(text, out_type=str)
        max_len = 0
        for sentence in tokens:
            max_len = max(max_len, len(sentence))
        sentence_length = max_len + 2
        # assert sentence_length <= max_length
        # sentence_length = max_length
        # max_len = sentence_length - 2
        ids = self.convert_tokens_to_ids(tokens)
        for sentence in ids:
            if len(sentence) < max_len:
                sentence += [self.pad_token_id] * (max_len - len(sentence))
            sentence += [self.eos_token_id, self.src_lang_id]
        ids = np.array(ids)
        attn_mask = np.ones([batch_size, sentence_length], dtype=np.int)
        if ref is None:
            return {"input_ids": ids, "attention_mask": attn_mask}
        else:
            tokens = self.sp_model.Encode(ref, out_type=str)
            max_len = 0
            for sentence in tokens:
                max_len = max(max_len, len(sentence))
            ids1 = self.convert_tokens_to_ids(tokens)
            for sentence in ids1:
                if len(sentence) < max_len:
                    sentence += [self.pad_token_id] * (max_len - len(sentence))
                sentence += [self.eos_token_id, self.tgt_lang_id]
            return {"input_ids": ids, "attention_mask": attn_mask, "labels": np.array(ids1)}

    def decode(self, input_ids: Union[M.Tensor, List[M.Tensor]]) -> Union[str, List[str]]:
        if isinstance(input_ids, M.Tensor):
            if input_ids.dim() == 2:
                return [self.convert_seperated_piece_to_natural_language(
                    self.remove_special_tokens(
                        self.convert_ids_to_tokens(
                            sentence
                        )
                    )
                ) for sentence in input_ids]
            if input_ids.dim() == 1:
                return self.convert_seperated_piece_to_natural_language(
                    self.remove_special_tokens(
                        self.convert_ids_to_tokens(
                            input_ids
                        )
                    )
                )
        elif isinstance(input_ids, list):
            return [self.convert_seperated_piece_to_natural_language(
                self.remove_special_tokens(
                    self.convert_ids_to_tokens(
                        sentence
                    )
                )
            ) for sentence in input_ids]
        raise ValueError(f"input_ids's dim must be 1 or 2, but given {input_ids.dim()}")

    def remove_special_tokens(self, sentence: List[str]) -> List[str]:
        ret = []
        for token in sentence:
            if token not in self.fairseq_tokens_to_ids:
                ret.append(token)
            if token == self.eos_token:
                break
        return ret

    def convert_seperated_piece_to_natural_language(self, sentence: List[str]) -> str:
        return self.sp_model.Decode(sentence)

    @property
    def vocab_size(self):
        return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1  # Plus 1 for the mask token

    @property
    def src_lang_id(self) -> int:
        return self.convert_tokens_to_ids(self.src_lang)

    @property
    def tgt_lang_id(self) -> int:
        return self.convert_tokens_to_ids(self.tgt_lang)

    @property
    def unk_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def eos_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def pad_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.pad_token)
