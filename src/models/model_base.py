#
# PRINCIPLES
# go line by line of references used - understand - re-write
# next time re-implement without reference and expand
# consistent - fast - sync - do 1 and 2 -- daily.
#

#
# script for base and associated model definitions
#

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from argparse import Namespace
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig # check what this does, use 'prepare_for_inference_'


#class BaseFairseqModel(nn.Module): #from models/model_base.py

class ModelBase(nn.Module):
    """Base class for any model, that is encoder-decoder, encoder/decoder only.
    Note: bunch of methods, ported from fairseq. Simplify/restrcture/remove at the end.
    """
    def __init__(self):     # check: why we need it.
        super().__init__()

    # removed classmethod add_args

    @classmethod
    def build_model(cls, args, task):
        """Implement this method to build a new model instance"""
        raise NotImplementedError

    def get_targets(self, sample, net_output):  # check: which script needs it, remove?
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):  # check: called whenever norm logprobs are needed. Do we need it for nmt, st? can we move this to other scripts?
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):     # check: are there a better approach, faster, simplified ? replace.
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def extract_features(self, *args, **kwargs): # check: two important methods but check necessity.
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def set_num_updates(self, num_updates): # check: simplify
        """State from trainer to pass along to model at every update."""
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

    def prepare_for_inference_(self, cfg: DictConfig): # check: simplify, avoid unnecessary after deepdive in generators.
        """Prepare model for inference."""
        kwargs = {}
        kwargs["beamable_mm_beam_size"] = (
            None
            if getattr(cfg.generation, "no_beamable_mm", False)
            else getattr(cfg.generation, "beam", 5)
        )
        kwargs["need_attn"] = getattr(cfg.generation, "print_alignment", False)
        if getattr(cfg.generation, "retain_dropout", False):
            kwargs["retain_dropout"] = cfg.generation.retain_dropout
            kwargs["retain_dropout_modules"] = cfg.generation.retain_dropout_modules
        self.make_generation_fast_(**kwargs)

    # Note: skipped legacy inference related method (make_generation_fast_), 'from_pretraind' methods.
    # e.g. 'load_state_dict' and associated methods,
    # since we don't need to upgrade *stat_dicts* assuming old ckpts,
    # opting to use the method 'load_state_dict' in 'nn.Module'.


