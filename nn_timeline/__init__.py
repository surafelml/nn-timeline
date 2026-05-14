"""
nn_timeline — Neural network architectures through time.

A minimal educational package tracing neural network architectures
from RNN through the Transformer era — versioned milestones, notes-first.

  pip install nn-timeline

Stable imports:
  from nn_timeline.archs.rnn         import LSTM, GRU, Seq2SeqRNN
  from nn_timeline.archs.tnn         import Transformer, GPT
  from nn_timeline.layers.attention  import MultiHeadAttention, BahdanauAttention
  from nn_timeline.layers.embeddings import SinusoidalPE, RoPE
  from nn_timeline.layers.norm       import RMSNorm
  from nn_timeline.layers.ffn        import SwiGLU
  from nn_timeline.layers.recurrent  import LSTMCell, GRUCell
  from nn_timeline.train             import Trainer
  from nn_timeline.metrics           import BLEU, chrF, Perplexity
"""

__version__ = "0.1.0-dev"
