"""FFN Base class and variants, assumed to use this in FFN specific applications/demos too"""

import torch
import torch.nn as nn

class FeedForwardNetworks(nn.Modules):
    """
    defines base and variants of FFN's
    module already implemented in 'arch_tnn_encoder/decoder'
    def build_fc1/2:

    TODO: move here for better clarity and variation.
    """
    raise NotImplementedError
