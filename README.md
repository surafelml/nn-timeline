# nn_timeline

A minimal package tracing neural network architectures and models through time.

```
pip install nn-timeline
```

## Motivation

The field moves fast, but understanding *why* each architecture superseded the last
requires slowing down. `nn_timeline` is built around that premise: each milestone
is a versioned, self-contained snapshot — RNN, Attention, Transformer — with
notes that trace the scientific reasoning, not just the implementation. The moat
is science depth and timeline; everything else is secondary.

For architectures beyond the Transformer era (MoE, alignment, etc.),
dedicated `nn_timeline_xyz` packages inherit this core and explore those
directions independently.

## What it is

- **Architecture timeline**: RNN → Attention → Transformer (versioned milestones, not a moving target)
- **Importable package**: `from nn_timeline.layers import RoPE, MultiHeadAttention`
- **Notes-first**: Quarto `.qmd`, math and derivations before every line of code
- **Runnable on Mac M\* or a single GPU** — no cluster required

## How it works

### Development loop

Every component starts with a science note — no code before the note exists.

```
write science note (.qmd)        ← always start here
  math, derivations, diagrams
         │
         ▼
write test (red)
         │
         ▼
implement until green
         │
         ▼
revise note with findings
         │
         ▼
tag versioned milestone
```

### Ecosystem: where new ideas go

`nn_timeline` is the stable baseline floor, not a prototyping surface.
When a new paper drops, the workflow is:

```
new paper (arxiv)
       │
       ├── need a baseline? ──► nn_timeline has it — import directly
       │
       ▼
create nn-timeline-{topic}
(separate repo, inherits nn_timeline)
       │
       ▼
write science note first (.qmd)
       │
       ▼
implement & benchmark vs nn_timeline baseline
       │
       ▼
foundational to the timeline? ──yes──► candidate for nn_timeline v{next}
                               ──no───► stays in nn-timeline-{topic}
```

This keeps `nn_timeline` stable and deep; extensions stay isolated until proven.

## Quick start

```python
from nn_timeline.archs.tnn import Transformer
from nn_timeline.archs.rnn import Seq2SeqRNN
from nn_timeline.layers.attention import BahdanauAttention, MultiHeadAttention
from nn_timeline.layers.embeddings import SinusoidalPE, RoPE
from nn_timeline.train import Trainer
from nn_timeline.metrics import BLEU
```

## Repository layout

```
nn_timeline/        # installable package
  archs/rnn/        # LSTM, GRU, Seq2SeqRNN
  archs/tnn/        # Transformer, GPT
  layers/attention/ # BahdanauAttention, MultiHeadAttention
  layers/embeddings/# SinusoidalPE, RoPE
  layers/ffn/       # FFN, SwiGLU
  layers/norm/      # LayerNorm, RMSNorm
  layers/recurrent/ # LSTMCell, GRUCell
  train/            # Trainer, optimizer, scheduler, checkpoint
  generate/         # beam search, sampling, kv_cache
  metrics/          # BLEU, chrF, Perplexity
  data/             # dictionary, BPE tokenizer, datasets
cli/                # preprocess / train / generate / evaluate
notes/              # Quarto .qmd — science notes, math-first, one concept per file
tests/              # pytest suite, TDD red→green
```

## Notes

Notes live at [surafelml.github.io/nn-timeline](https://surafelml.github.io/nn-timeline) (Quarto → GitHub Pages).
Each notebook imports from `nn_timeline` directly — code is always the SSOT.

## Pre-trained models

Checkpoints are hosted on [HuggingFace Hub](https://huggingface.co/surafelml).

```python
from nn_timeline.train.checkpoint import load_from_hub
model = load_from_hub("surafelml/transformer-mt-en-de-small")
```

## Ecosystem

All repos live under [`github.com/surafelml`](https://github.com/surafelml) with a shared `nn-timeline` prefix.

| Repo | Purpose |
|---|---|
| [`nn-timeline`](https://github.com/surafelml/nn-timeline) (this) | Core architectures, RNN → Transformer era |
| [`nn-timeline-ui`](https://github.com/surafelml/nn-timeline-ui) | Gradio demos, interactive timeline, experiment dashboard |
| `nn-timeline-{moe,alignment,…}` | Architecture-specific extensions |

## License

MIT
