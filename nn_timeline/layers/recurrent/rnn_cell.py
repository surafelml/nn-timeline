"""Notes: notes/01_rnn/01_rnn_bptt.qmd"""
import math

import torch
import torch.nn as nn


class RNNCell(nn.Module):
    """Vanilla RNN cell: h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # tanh' <= 1, so repeated composition shrinks gradients with depth —
        # the vanishing-gradient mechanism 05_gradient_flow.qmd builds on.
        return torch.tanh(
            x @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
        )
