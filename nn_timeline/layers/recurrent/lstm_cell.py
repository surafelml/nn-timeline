"""Notes: notes/01_rnn/02_lstm_gru.qmd"""
import math
from typing import Tuple

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """LSTM cell with input/forget/cell/output gates (Hochreiter & Schmidhuber 1997).

    Gate order in the stacked weight/bias tensors matches
    ``torch.nn.LSTMCell``: input, forget, cell (candidate), output.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gi = x @ self.weight_ih.T + self.bias_ih
        gh = h @ self.weight_hh.T + self.bias_hh
        i, f, g, o = (gi + gh).chunk(4, dim=-1)

        i = torch.sigmoid(i)   # input gate: how much of the candidate to write
        f = torch.sigmoid(f)   # forget gate: how much of the old cell state to keep
        g = torch.tanh(g)      # candidate cell content
        o = torch.sigmoid(o)   # output gate: how much of the cell state to expose as h

        # additive c1 (not multiplicative like plain RNN's tanh composition) is the
        # constant-error-carousel fix for vanishing gradients — 02_lstm_gru.qmd's core claim.
        c1 = f * c + i * g
        h1 = o * torch.tanh(c1)
        return h1, c1
