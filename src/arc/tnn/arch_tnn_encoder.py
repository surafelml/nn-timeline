"""Defines the self-attention encoder in [cite-paper]"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional
from src.utils.helpers import get_activation_fn
from src.models.model_base_encoder import ModelBaseEncoder
from src.layers.shared import Dropout, LayerNorm, LayerDropModuleList
from src.layers.embeddings import PositionalEmbedding
from src.layers.attention import MultiHeadSelfAttention
import math


"""
Script consistitue two main ...
- layer blocks for the model defn - potentially move this to layers module to make it accessible for other model defns.
- layer definition/model ?
"""

class ArchTnnEncoder(ModelBaseEncoder):
    """Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
        is a :class:`TransformerEncoderLayer`.
        Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens):
        self.cfg = cfg # check: cfg comes from transformer_config dataclass
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3])) # check: ?

        self.dropout_module = Dropout(cfg.dropout)
        self.encoder_layerdrop = cfg.encoder.layerdrop # check, reduce transformer depth with layer dropout

        embed_dim = embed_tokens.embedding_dim  # check: assignment origin ?
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = cfg.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)  # check: link paper section

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.encoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        # if not cfg.adaptive_input and cfg.quant_noise.pq > 0: # check: removing quant_noise paper implementation for now
        #     self.quant_noise = apply_quant_noise_(
        #         nn.Linear(embed_dim, embed_dim, bias=False),
        #         cfg.quant_noise.pq,
        #         cfg.quant_noise.pq_block_size,
        #     )
        # else:
        #     self.quant_noise = None

        if self.encoder_layerdrop > 0.0: # check: keeping this paper implementation.
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None


    def build_encoder_layer(self, cfg):
        # FIXME: specifc layer defn are in ./layers/attention, the skeleton/arch is defined here, overridding
        #layer = transformer_layer.TransformerEncoderLayerBase(cfg)
        layer = TnnEncoderLayer(cfg) # calls encoder block for tnn, actual layers are defined in ./layers/*

        # checkpoint = cfg.checkpoint_activations   # check: important for data parallel, address later!
        # if checkpoint:
        #     offload_to_cpu = cfg.offload_activations
        #     layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        #
        # # if we are checkpointing, enforce that FSDP always wraps the
        # # checkpointed layer, regardless of layer size
        # min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        # layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)

        return layer

    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)

        return x, embed

    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)




class TnnEncoderLayer(nn.module):  # check: in fairseq: modules/TransformerEncoderLayerBase
    """Encoder layer block.

        In the original paper each operation (multi-head attention or FFN) is
        postprocessed with: `dropout -> add residual -> layernorm`. In the
        tensor2tensor code they suggest that learning is more robust when
        preprocessing each layer with layernorm and postprocessing with:
        `dropout -> add residual`. We default to the approach in the paper, but the
        tensor2tensor approach can be enabled by setting
        *cfg.encoder.normalize_before* to ``True``.

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder.embed_dim

        #self.quant_noise = cfg.quant_noise.pq
        #self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.self_attn = self.build_self_attention(self.embed_dim, cfg)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.dropout_module = Dropout(cfg.dropout) #,module_name=self.__class__.__name__)
        # TODO: we need module name specially for decoding time.

        # check: re-do activation call and layer simplification
        self.activation_fn = get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = Dropout(float(activation_dropout_p))
        self.normalize_before = cfg.encoder.normalize_before

        # FIXME: FFN is missing b/c certain ops are moved to quantization wrapper (Liner, Embedding, Conv2d)
        # DIFF: https://github.com/pytorch/fairseq/commit/1c8ab79ca59b466120e3df448673cab840f571ea#diff-f0eb88613a42eb9195c95be7d629fcabaeb7d451f8fe16587de03c433eb300d9
        # TODO: reverse back (FFN, SA Liner layer ops) to pre quant_noise (which is used for model compression) for clarity.
        #   use proper naming. why not define FFN as in OpeNMT-tf, dense net?
        self.fc1 = self.build_fc1(self.embed_dim, cfg.encoder.ffn_embed_dim)    # input for ffn
        self.fc1 = self.build_fc2(self.encoder.ffn_embed_dim, self.embed_dim)   # output for ffn
        # self.fc1 = self.build_fc1(
        #     self.embed_dim,
        #     cfg.encoder.ffn_embed_dim,
        #     self.quant_noise,
        #     self.quant_noise_block_size,
        # )
        # self.fc2 = self.build_fc2(
        #     cfg.encoder.ffn_embed_dim,
        #     self.embed_dim,
        #     self.quant_noise,
        #     self.quant_noise_block_size,
        # )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)
    # def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
    #     return quant_noise(
    #         nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
    #     )
    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)
    # def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
    #     return quant_noise(
    #         nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
    #     )

    def build_self_attention(self, embed_dim, cfg):
        return MultiHeadSelfAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            #q_noise=self.quant_noise,
            #qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    # check: removed update of state dict

    def forward(
            self,
            x,
            encoder_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        # begin passing x (input) through the sub-layer modules
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # object is not callable - why?
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # correlate this steps with the transformer figure/equations
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        # check: FFN of openmt-tf which is way cleaner.
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x    # rename x with proper naming



class TnnEncoderLayerXYZ(TnnEncoderLayer):
    """
    Placeholder to define new variant of the self-attention layers, parenting the standard self-attention.
    """
    raise NotImplementedError

