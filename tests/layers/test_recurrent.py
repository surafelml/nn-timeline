"""T1.1 acceptance criteria: RNNCell, LSTMCell, GRUCell."""
import inspect

import pytest
import torch

from nn_timeline.layers.recurrent import RNNCell, LSTMCell, GRUCell

BATCH, INPUT_SIZE, HIDDEN_SIZE = 8, 12, 16

AVAILABLE_DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    AVAILABLE_DEVICES.append("mps")
if torch.cuda.is_available():
    AVAILABLE_DEVICES.append("cuda")


def _seeded(*shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g)


class TestRNNCell:
    def test_output_shape(self):
        cell = RNNCell(INPUT_SIZE, HIDDEN_SIZE)
        x = _seeded(BATCH, INPUT_SIZE)
        h0 = _seeded(BATCH, HIDDEN_SIZE)
        h1 = cell(x, h0)
        assert h1.shape == (BATCH, HIDDEN_SIZE)

    def test_matches_pytorch_reference(self):
        ref = torch.nn.RNNCell(INPUT_SIZE, HIDDEN_SIZE)
        mine = RNNCell(INPUT_SIZE, HIDDEN_SIZE)
        mine.weight_ih.data.copy_(ref.weight_ih.data)
        mine.weight_hh.data.copy_(ref.weight_hh.data)
        mine.bias_ih.data.copy_(ref.bias_ih.data)
        mine.bias_hh.data.copy_(ref.bias_hh.data)

        x, h0 = _seeded(BATCH, INPUT_SIZE), _seeded(BATCH, HIDDEN_SIZE)
        h_ref = ref(x, h0)
        h_mine = mine(x, h0)
        assert torch.allclose(h_ref, h_mine, atol=1e-5)

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_runs_on_available_devices(self, device):
        cell = RNNCell(INPUT_SIZE, HIDDEN_SIZE).to(device)
        x = _seeded(BATCH, INPUT_SIZE).to(device)
        h0 = _seeded(BATCH, HIDDEN_SIZE).to(device)
        h1 = cell(x, h0)
        assert h1.device.type == device

    def test_notes_reference_header(self):
        source = inspect.getsource(inspect.getmodule(RNNCell))
        assert "01_rnn/01_rnn_bptt.qmd" in source


class TestGRUCell:
    def test_output_shape(self):
        cell = GRUCell(INPUT_SIZE, HIDDEN_SIZE)
        x = _seeded(BATCH, INPUT_SIZE)
        h0 = _seeded(BATCH, HIDDEN_SIZE)
        h1 = cell(x, h0)
        assert h1.shape == (BATCH, HIDDEN_SIZE)

    def test_matches_pytorch_reference(self):
        ref = torch.nn.GRUCell(INPUT_SIZE, HIDDEN_SIZE)
        mine = GRUCell(INPUT_SIZE, HIDDEN_SIZE)
        mine.weight_ih.data.copy_(ref.weight_ih.data)
        mine.weight_hh.data.copy_(ref.weight_hh.data)
        mine.bias_ih.data.copy_(ref.bias_ih.data)
        mine.bias_hh.data.copy_(ref.bias_hh.data)

        x, h0 = _seeded(BATCH, INPUT_SIZE), _seeded(BATCH, HIDDEN_SIZE)
        h_ref = ref(x, h0)
        h_mine = mine(x, h0)
        assert torch.allclose(h_ref, h_mine, atol=1e-5)

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_runs_on_available_devices(self, device):
        cell = GRUCell(INPUT_SIZE, HIDDEN_SIZE).to(device)
        x = _seeded(BATCH, INPUT_SIZE).to(device)
        h0 = _seeded(BATCH, HIDDEN_SIZE).to(device)
        h1 = cell(x, h0)
        assert h1.device.type == device

    def test_notes_reference_header(self):
        source = inspect.getsource(inspect.getmodule(GRUCell))
        assert "01_rnn/02_lstm_gru.qmd" in source


class TestLSTMCell:
    def test_output_shape(self):
        cell = LSTMCell(INPUT_SIZE, HIDDEN_SIZE)
        x = _seeded(BATCH, INPUT_SIZE)
        h0, c0 = _seeded(BATCH, HIDDEN_SIZE), _seeded(BATCH, HIDDEN_SIZE)
        h1, c1 = cell(x, (h0, c0))
        assert h1.shape == (BATCH, HIDDEN_SIZE)
        assert c1.shape == (BATCH, HIDDEN_SIZE)

    def test_matches_pytorch_reference(self):
        ref = torch.nn.LSTMCell(INPUT_SIZE, HIDDEN_SIZE)
        mine = LSTMCell(INPUT_SIZE, HIDDEN_SIZE)
        mine.weight_ih.data.copy_(ref.weight_ih.data)
        mine.weight_hh.data.copy_(ref.weight_hh.data)
        mine.bias_ih.data.copy_(ref.bias_ih.data)
        mine.bias_hh.data.copy_(ref.bias_hh.data)

        x = _seeded(BATCH, INPUT_SIZE)
        h0, c0 = _seeded(BATCH, HIDDEN_SIZE), _seeded(BATCH, HIDDEN_SIZE)
        h_ref, c_ref = ref(x, (h0, c0))
        h_mine, c_mine = mine(x, (h0, c0))
        assert torch.allclose(h_ref, h_mine, atol=1e-5)
        assert torch.allclose(c_ref, c_mine, atol=1e-5)

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_runs_on_available_devices(self, device):
        cell = LSTMCell(INPUT_SIZE, HIDDEN_SIZE).to(device)
        x = _seeded(BATCH, INPUT_SIZE).to(device)
        h0 = _seeded(BATCH, HIDDEN_SIZE).to(device)
        c0 = _seeded(BATCH, HIDDEN_SIZE).to(device)
        h1, c1 = cell(x, (h0, c0))
        assert h1.device.type == device
        assert c1.device.type == device

    def test_notes_reference_header(self):
        source = inspect.getsource(inspect.getmodule(LSTMCell))
        assert "01_rnn/02_lstm_gru.qmd" in source
