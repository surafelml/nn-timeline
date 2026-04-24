"""Transformer model as in: paper"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from src.models.encoder_decoder import EncoderDecoder
from src.archs.tnn.arch_tnn_encoder import ArchTnnEncoder
from src.archs.tnn.arch_tnn_decoder import ArchTnnDecoder
from torch import Tensor
from src.layers.shared import Embedding
from src.configs.configs import TransformerConfig
from src.configs.dataclass_parser import gen_parser_from_dataclass


# TODO: CLEAN - FOCUS ON EACH PIECE


# FIXME: add model register or find another solution
class ArchTnnBase(EncoderDecoder): # TnnBase ? why to add arch_ on all as the folders says it
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        #self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        # TODO: instead of populating it here as legacy,
        #   simplify args parsing, avoid the complexity of of using utils/gen_parser_from_dataclass,
        #   dataclass TransformerConfig, see opennmt-tf config fo example. This might need more time...General layout:
        #   use arch specific dataclasses (like auto_config and from file/yaml)
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser,
            TransformerConfig(),
            delete_default=False,
            with_prefix="")

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        # check: disable loading emb files, move this to future task.
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        # if path:
        #     embed_dict = utils.parse_embedding(path)
        #     utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return ArchTnnEncoder(cfg, src_dict, embed_tokens)  # TODO: encoder

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):    # TODO: decoder
        return ArchTnnDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,        # check: paper, move to future task.
        )

    @classmethod
    def build_model(cls, cfg, task): # TODO: commented out some configs. Requires more cleaning and simplify.
        """Build a new model instance.
        Setting parameters.
        """

        # --  TODO T96535332 # check: latest fairseq if this addressed.
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        # check: removed, its for paper on dropping layers, move to future paper impl.
        # if cfg.encoder.layers_to_keep:
        #     cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        # if cfg.decoder.layers_to_keep:
        #     cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                    cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )


        if cfg.offload_activations:     # check: if necessary to keep
            cfg.checkpoint_activations = True  # offloading implies checkpointing

        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)

        # if not cfg.share_all_embeddings:  # check: its for data parallel, check necessity, if yes simplify and organize.
        #     # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
        #     encoder = fsdp_wrap(encoder, min_num_params=cfg.min_params_to_wrap)
        #     decoder = fsdp_wrap(decoder, min_num_params=cfg.min_params_to_wrap)

        return cls(cfg, encoder, decoder)

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        NB: this is the type of comments/notes we need >>>
            # TorchScript doesn't support optional arguments with variable length (**kwargs).
            # Current workaround is to add union of all arguments in child classes.

        For instance, if we re-write this help as descriptive as possible:
        This method runs the forward pass for and encoder-decoder model. Describe args and returns.
        Why specific approach is followed like torchscript not using **kwargs.
        """

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

