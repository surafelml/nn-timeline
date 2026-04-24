"""
Train model scripts - exposes the train task api
TODO: why not import all interfaces and expose them with run type
    [preprocess, train, translate]
"""

from src.tasks.train import cli_main

if __name__ == "__main__":
    cli_main