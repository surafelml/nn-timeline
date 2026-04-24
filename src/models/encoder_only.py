import torch
import torch.nn.functional as F
from src.models.model_base import ModelBase
from src.models.model_base_encoder import ModelBaseEncoder

class Encoder(ModelBase):
    """
    Base class for encoder-only models.
    Used in tasks such as masked LM pre-training, and ASR.
    """

    def __init__(self, encoder):
        """
        :param encoder: EncoderBase
        """
        super().__init__()
        self.encoder = encoder
        isinstance(self.encoder, ModelBaseEncoder)

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        Run the forward pass for an encoder-only model.

        Feeds a batch of tokens through the encoder to generate features.

        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the encoder's output, typically of shape `(batch, src_len, features)`
        """
        return self.encoder(src_tokens, src_lengths, **kwargs)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """
        Get normalized probabilities (or log probs) from a net's output.
        :param net_output:
        :param log_probs:
        :param sample:
        :return:
        """
        encoder_out = net_output["encoder_out"]
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions()