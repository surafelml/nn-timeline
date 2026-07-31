"""Notes: notes/01_rnn/02_lstm_gru.qmd"""
import math

import torch
import torch.nn as nn


class GRUCell(nn.Module):
    """GRU cell with reset/update gates (Cho et al. 2014).

    Gate order in the stacked weight/bias tensors matches
    ``torch.nn.GRUCell``: reset, update, new.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gi = x @ self.weight_ih.T + self.bias_ih
        gh = h @ self.weight_hh.T + self.bias_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        r = torch.sigmoid(i_r + h_r)   # reset gate: how much of h to use for the candidate
        z = torch.sigmoid(i_z + h_z)   # update gate: candidate vs. carry-over mix
        n = torch.tanh(i_n + r * h_n)  # candidate state, gated by r
        # z -> 1 carries h through unchanged, z -> 0 replaces it with the candidate —
        # GRU's single-gate analogue of the LSTM's separate forget/input gates.
        return (1 - z) * n + z * h
