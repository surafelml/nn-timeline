"""
Defines attention, multi-head attention, and associated modules of attention (such as reduction)
"""


import torch
import torch as pt  # todo: proper way to load torch in sitmt
import torch.nn as nn
from torch import Tensor
from src.layers.shared import Dropout
import torch.nn.functional as FUNC

import math
from typing import Dict, Optional, Tuple


# Add description
# Copyright and modification copyright

class BaseAttention(nn.Modules):
    """Basic attention variant as in [cite-paper]
    Can compute similar attention using MultiHeadSelfAttention
    with arg=encoder=attention."""

    raise NotImplementedError


# TODO: re-do the full mha following fairseq, simplify/add variants after first working version.
# called by arch_tnn_*
class MultiHeadSelfAttention(nn.Modules):
    """Implements multi-head attention as in [cite-paper]
    Should Implement every single component or copy some full scripts here and avoid the deep dive?
    Timing: ? if going full gone take ?

    """

    def __init__(self,
                embed_dim,
                num_heads,
                kdim=None,
                vdim=None,
                dropout=0.0,
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False,
                self_attention=False,
                encoder_decoder_attention=False):
        """
        ...
        """

        super().__init__()
        #self.model_dim = model_dim
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        # self.attn_heads = attn_heads
        self.num_heads = num_heads
        # model dimensions
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        # TODO: set dropout for module/net layer specific (fairseq/modules/fairseq_dropout)
        self.attn_dropout = dropout


        # attention type
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size")


        # Q: why do we have to compute different linear projections ?
        # projections
        self.linear_query = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.linear_key = nn.Linear(self.kdim, self.embed_dim, bias=bias)
        self.linear_value = nn.Linear(self.vdim, self.embed_dim, bias=bias)

        self.linear_output = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)


        # TODO: can add build for relative PE https://github.com/OpenNMT/OpenNMT-tf/blob/5b7e42c653ffbec90b44d71bb9c6a0fc9a3816d2/opennmt/layers/transformer.py#L270

        from torch.nn import Parameter
        # add bias - identify which task needs this, following principle to remove antying not in base transformer.
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        # identify which tasks requires it
        self.add_zero_attn = add_zero_attn
        self.reset_parameters() # there is a method definition in fairseq mult-head attention


   #def forward(self, input, memory, mask=None, incremental_state=None):
    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
            TODO: does not implement incremental_state in v1.0
        """
        if need_head_weights:
            need_weights = True

        # dvice type
        #is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = input.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # key, value comes from encoder (in standard enc-dec attn), else from prv layer
        # TODO: why key, value is passed to forward in fairseq, why not compute from memory?
        #   memory being the sequence to attend
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if (
                incremental_state is None
                and not static_kv
                # A workaround for quantization to work. Otherwise JIT compilation
                # treats bias in linear module as method.
                and not torch.jit.is_scripting()
        ):
            assert key is not None and value is not None

            # TODO: see pytorch F.multi_head_.. that implements a 300 line of code, re-implement custom to call here
            return FUNC.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
            # returns the attention, and attention weights optionally


        # TODO: simplifying two ways forward pass for multi-head attn & impl. multi-head instead of using the tnn_fun
        # i. no-incremental: state (prv time stamp is not cached)
        # ii. incremental: with/without recomputing key and value
        # i. Defn non-incremental here using F.multi_head_attention_forward
        if incremental_state is None:
            raise NotImplementedError
            # see https://github.com/pytorch/fairseq/issues/166 about incremental decoder introduction
            # i. use previously decoded output (input-feeding?), and ii. their long term hidden representation




        # REQUIERS SOME DEEP DIVE AND A DAY OF WORK OFR QaD approach
        #     assert key is not None and value is not None
        #     return tnn_fun.multi_head_attention_forward(
        #         query,
        #         key,
        #         value,
        #     )
        #
        #
        # # query comes from the target, why not use 'memory'
        # query = self.linear_query(query)
        # key = self.linear_key(query)
        # value =  self.linear_value(query)
        #
        #
        # # ii. incremental with self-attention or decoder-encoder attention
        # # do linear projects
        # if self.comp_self_attention:
        #     # self attention within encoder or decoder
        #     q = self.linear_query(query)
        #     k = self.linear_key(query)
        #     v = self.linear_value(query)
        # else: # compute encoder-decoder cross attention
        #     q = self.linear_query(query)
        #     if key is None:
        #         assert value is None
        #         k = v = None
        #     else:
        #         k = self.linear_key(key)
        #         v = self.linear_value(key)
        #
        # # Comp: dot prod of q with all keys, divide each by sqrt(d_k), apply softmax to get weights of the values
        # self.scaling = self.model_dim ** -0.5 # 1/sqrt(model_dim)
        # q *= self.scaling # what is K^T, see concat
        #
        #
        #
        # # Placeholder (PL): fairseq allows to apply bias to k-v pairs
        #
        #
        # # concat with .view before final linear pojection. Separate from scaling, b/c k might be None ?
        # q = (
        #     q.contiguous()
        #     .view(tgt_len, bsz * self.num_heads, self.head_dim)
        #     .transpose(0, 1)
        # )
        # if k is not None:
        #     k = (
        #         k.contiguous()
        #         .view(-1, bsz * self.num_heads, self.head_dim)
        #         .transpose(0, 1)
        #     )
        # if v is not None:
        #     v = (
        #         v.contiguous()
        #         .view(-1, bsz * self.num_heads, self.head_dim)
        #         .transpose(0, 1)
        #     )
        #
        #
        # # call_dropout = Dropout.dropout(attn, self.attn_dropout, training=True)
        # #FIXME: rest of multi-head [L:261-], it really requires simplification as in the paper, modularizing the rest.

