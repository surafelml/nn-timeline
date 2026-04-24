"""Defines shared functionalities across layers and/or modules

    NB: this shared can split dropouts, normalization, etc if getting complicated.
"""
import torch
import torch.nn as pt_nn
import torch.nn.functional as pt_nn_func
import logging
from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def Embedding(num_embeddings, embedding_dim, padding_idx): # check: find a better place
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m



class Dropout(pt_nn.Modules):
    """Main class for any dropout/s applied on layers
    Paper/s: https://arxiv.org/pdf/1207.0580.pdf
    """
    # def __init__(self, dropout_prob):
    #     self.dropout_prob = dropout_prob
    #     # TODO: inference time and dropout variant configs
    #
    # def dropout(self, input, training=False, inplace=False):
    #     """Base dropout only at training time"""
    #     if self.dropout_prob > 0 and training:
    #         return pt_nn_func.dropout(input, p=self.dropout_prob, training=training, inplace=inplace)
    #     else:
    #         return input

    def __init__(self, p): #, module_name=None):
        super().__init__()
        self.p = p
        # self.module_name = module_name    # check: used for 'make_generation_fast_'
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

class LayerDropModuleList(nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m



# class LayerNormalization(pt_nn.Modules):
#     """Main class for layer normalization
#     Paper/s:
#
#     """
try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)



class LayerPrePostProcess(pt_nn.Modules):
    """Main class for preprocessing and post processing layer input and output
    Paper/s:

    """
    raise NotImplementedError



"""Activations: should go to utils/activations, but important for layers."""
def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


#def activation_functions(activation: str) -> Callable:
#    raise NotImplementedError
# see helpers.py