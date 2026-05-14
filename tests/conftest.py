import torch
import pytest


@pytest.fixture
def synthetic_batch():
    """8 sentence pairs, src/tgt length 10, vocab 50 — CPU-only, no real data."""
    src = torch.randint(1, 50, (8, 10))
    tgt = torch.randint(1, 50, (8, 10))
    return src, tgt
