from src.models.model_base import ModelBase
from src.models.model_base_encoder import ModelBaseEncoder
from src.models.model_base_decoder import ModelBaseDecoder


class EncoderDecoder(ModelBase):
    """Encoder-Decoder model definition that inherits ModelBase.
    Can be inherited when modeling seq2seq architectures (e.g. NMT using TNN's).
    Raises NotImplementedError for methods sub-class should implement.
    See selective description under each method for further understanding of class ops.
    """

    def __init__(self, encoder, decoder):
        """
        :param encoder: encoder module.
        :param decoder: decoder module.
        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert isinstance(self.encoder, ModelBaseEncoder)
        assert isinstance(self.decoder, ModelBaseDecoder)

    # check: if this simplification is ok (adding methods for encoder-decoder model) for inheriting classes, or necessary?
    def encode(self, src_tokens, src_lengths, **kwargs): # call the encoder
        return self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

    def decode(self, pre_output_tokens, output_of_encoder, **kwargs): # call the decoder
        return self.decoder(pre_output_tokens, output_of_encoder, **kwargs)

    def forward(self, src_tokens, src_lengths, pre_output_tokens, **kwargs):
        """
        Runs the forward pass for a seq2seq arch...
        forward pass takes the encoder input

        Args and Returns:
        :param src_tokens:
        :param src_lengths:
        :param pre_output_tokens: applies teacher forcing using previous decoder output/s
        :param kwargs:
        :return:
        """

        output_of_encoder = self.encode(src_tokens, src_lengths, **kwargs)
        output_of_decoder = self.decode(pre_output_tokens, output_of_encoder, **kwargs)

        return output_of_decoder

    # check: avoided 'forward_decoder' which seem to have use in iterative_refinement_decoder
    # check: added methods below, see for simplification and necessity.

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.
        NB: Applies softmax on input from projection layer, outputs with vocab size.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()