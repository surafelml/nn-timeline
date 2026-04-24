
"""
Layers for the attention is all you need [insert-paper] model
"""


import torch
import torch.nn as nn


# define here non state preserving methods that are input for the classes

from .attention import MultiHeadSelfAttention
from .feedforward import FeedForwardNetworks
from .shared import LayerPrePostProcess # assuming this will include several options


class TNNEncoderLayer(nn.Module):
    """Self-Attention Encoder Layer stack definition - CONCISE.

    :arg

    """

    def __init__(self, args):
        self.self_attention = MultiHeadSelfAttention(

        )

        self.self_attention = LayerPrePostProcess(

        )

        self.feed_forward = FeedForwardNetworks(

        )

        self.feed_forward = LayerPrePostProcess(

        )

    def forward(self, x, mask_padding):
        """
        :param x: input, shape: seq_len X batch X emb_dim, type: Tensor
        :param mask_padding: mask input, shape: batch X seq_len, type: ByteTensor
        :return: encoder layer output, shape: seq_len X batch X emb_dim, type: ?
        TODO: attention_mask: check usage in fairseq before importing/defining here.
        """

        x, _ = self.self_attention(x, mask=mask_padding)
        x = self.feed_forward(x)

        return x


# FAIRSEQ - TF FOR CONCISE AND CLARITY CROSS REFERENCE - DEFINE SIT WAY


class TNNDecoderLayer(nn.Module):
    """Implements transformer decoder layer brining all the modules"""
