"""
Each script file should adhere to the three rules of my writing: concise, precise, simple.

Last one "simple" might not be easy if you are begining but from variable naming to the concepts encoded, there should
be some direction to it.

Optional/Now Must: simplification for multi- encoder-decoder model defnition, currently in model.py as a class
"""

import logging
from typing import Dict, List, Optional
import torch.nn as nn

from src.models.model_base import ModelBase
from src.models.model_base_encoder import ModelBaseEncoder
from src.models.model_base_decoder import ModelBaseDecoder
from src.models.encoder_decoder import EncoderDecoder
from src.data.dictionary import Dictionary



class EncoderDecoderMulti(ModelBase):
    """Similar to EncoderDecoder Model, class combines multiple encoder-decoder models.
    Notice:
    - different from a single encoder-decoder model, we use [key] to access each encoder-decoder properties.
    - additionally 'build_shared_embeddings' is introduced given the multiple vocabs per model directions.
    """

    def __init__(self, encoders, decoders):
        """
        :param encoders: encoders
        :param decoders: decoders
        """
        super().__init__()

        # check encoders and decoders type, create models instance
        assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            isinstance(encoders[key], ModelBaseEncoder)
            isinstance(decoders[key], ModelBaseDecoder)

        self.models = nn.ModuleDict(
            {
                key: EncoderDecoder(encoders[key], decoders[key])
                for key in self.keys
            }
        )

    @staticmethod
    def build_shared_embeddings(
            dicts: Dict[str, Dictionary],
            langs: List[str],
            embed_dim: int,
            build_embedding: callable,
            pretrained_embed_path: Optional[str] = None,
    ):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                "--share-*-embeddings requires a joined dictionary: "
                "--share-encoder-embeddings requires a joined source "
                "dictionary, --share-decoder-embeddings requires a joined "
                "target dictionary, and --share-all-embeddings requires a "
                "joint source + target dictionary."
            )
        return build_embedding(shared_dict, embed_dim, pretrained_embed_path)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        raise NotImplementedError

    @property
    def encoder(self):
        return self.models[self.keys[0]].encoder

    @property
    def decoder(self):
        return self.models[self.keys[0]].decoder

    def max_positions(self):
        """Maximum length supported by the model."""
        return {
            key: (
                self.models[key].encoder.max_positions(),
                self.models[key].decoder.max_positions(),
            )
            for key in self.keys
        }

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return min(model.decoder.max_positions() for model in self.models.values())

    # Note: as in model_encoder_decoder, left out 'load_state_dict' and 'forward_decoder'. Simplify further.
