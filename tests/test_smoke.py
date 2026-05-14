"""
Smoke tests — toy configs, CPU only, no HF Hub, no real data.
Every architecture must pass before CI is green.
"""
import nn_timeline


def test_import():
    assert nn_timeline.__version__ == "0.1.0-dev"
