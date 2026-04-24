# nn_timeline — Phase 1 Tickets

> Working board for Phase 1 (sitMT → nn_timeline, target: v1.0.0).
> Tickets are ordered by dependency tier — all tickets in a tier can be worked in parallel.
> Milestone: LSTM Seq2Seq trains on a toy parallel corpus end-to-end. That validates the full stack before anything else ships.

---

## Dependency Order

```
T0 (Setup)
  └─► T1 (Layers)  ──────────────────┐
        └─► T2 (Archs)               ├─► T4 (Train)
      T3 (Data + Device) ────────────┘     └─► T5 (Generate + Metrics)
                                                  └─► T6 (Tests + Milestone)
                                                        └─► T7 (Notes + Web)
                                                              └─► T8 (HF Hub)
```

---

## Tier 0 — Setup (no dependencies, start here)

### T0.1 — Repo restructure + package scaffold

Rename sitMT → nn_timeline. Create the full directory tree from Section 4 of
ssot_plan.md. Old code stays accessible on a `legacy/sitmt` branch.

**Includes:**
- Create all directories with `__init__.py` stubs
- Move existing sitMT files to their new locations (or park in `legacy/`)
- Verify `import nn_timeline` works from repo root

**Acceptance criteria:**
- `python -c "import nn_timeline"` succeeds with no errors
- Directory tree matches Section 4 exactly
- No broken imports from moved files
- `legacy/sitmt` branch preserves original code

---

### T0.2 — pyproject.toml + MIT LICENSE

Package installable as `pip install nn-timeline` (PyPI) and `pip install -e .` (dev).

**Includes:**
- `pyproject.toml` with name, version (`0.1.0-dev`), deps, entry points
- `LICENSE` (MIT)
- `.gitignore`, `README.md` stub with install instructions
- Confirm declared dependencies: `torch>=2.0`, `numpy`, `sentencepiece`

**Acceptance criteria:**
- `pip install -e .` succeeds on a fresh venv (Python 3.10+)
- `pip show nn-timeline` lists correct metadata
- No undeclared imports in the package surface
- `python -c "from nn_timeline.layers.attention import mha"` resolves (even if stub)

---

### T0.3 — `archs/timeline_registry.py`

Structured arch metadata store. Single source of truth for arch picker UI
(nn_timeline_ui) and notes cross-referencing.

**Includes:**
- Dict of entries, one per arch, with fields:
  `name`, `year`, `family`, `key_innovation`, `note_path`, `stable` (bool),
  `task` (list: lm / mt / mlm / s2t), `paper` (citation string)
- Covers all Phase 1 archs: RNN family + Transformer family
- Helper: `get_by_family(family)`, `get_stable()`, `get_by_task(task)`

**Acceptance criteria:**
- `from nn_timeline.archs.timeline_registry import REGISTRY` works
- All Phase 1 archs present with complete fields
- `get_by_family("rnn")` returns correct subset
- No external dependencies (pure Python dict)

---

## Tier 1 — Layers (parallel, after T0)

### T1.1 — Recurrent cells (`layers/recurrent/`)

RNNCell, LSTMCell, GRUCell. These are the atomic units the RNN archs compose.

**Includes:**
- `rnn_cell.py` — vanilla RNN: `h_t = tanh(W_ih * x + W_hh * h)`
- `lstm_cell.py` — LSTM: input/forget/cell/output gates
- `gru_cell.py` — GRU: reset/update gates
- All export from `nn_timeline.layers.recurrent`

**Acceptance criteria:**
- Output shapes correct: `(batch, hidden_dim)` for RNN/GRU, `(h, c)` for LSTM
- Runs on MPS, CUDA, CPU without code change
- Matches PyTorch built-in numerics within tolerance (test against `nn.RNNCell` etc.)
- Notes reference header present in each file

---

### T1.2 — Norm layers (`layers/norm/`)

LayerNorm and RMSNorm. Small but used everywhere — must be correct.

**Includes:**
- `layer_norm.py` — wraps or reimplements `nn.LayerNorm` with educational annotation
- `rms_norm.py` — `x / rms(x) * weight`, no bias (modern SLM standard)
- Export from `nn_timeline.layers.norm`

**Acceptance criteria:**
- `LayerNorm` output matches `torch.nn.LayerNorm` numerically
- `RMSNorm` output matches reference implementation (e.g., LLaMA's)
- Both handle arbitrary input shapes (batch, seq, dim)

---

### T1.3 — FFN layers (`layers/ffn/`)

Position-wise feedforward networks.

**Includes:**
- `feedforward.py` — standard FFN: `Linear → activation → Linear`, configurable activation
- `swiglu.py` — SwiGLU: `(xW) * silu(xV)` gate, used in LLaMA-style (Phase 1 stub, exercised in Phase 2)
- Export from `nn_timeline.layers.ffn`

**Acceptance criteria:**
- `FeedForward(dim=512, ffn_dim=2048)(x)` returns shape `(batch, seq, 512)`
- Activation configurable: ReLU, GELU, SiLU
- `SwiGLU` implemented and importable even if not exercised until Phase 2

---

### T1.4 — Attention layers (`layers/attention/`)

The full attention family across the timeline.

**Includes:**
- `bahdanau.py` — additive attention: `score = v * tanh(W1*h + W2*s)`
- `luong.py` — multiplicative (dot / general / concat variants)
- `mha.py` — multi-head self-attention + cross-attention (Vaswani 2017)
- `gqa.py` — grouped query attention stub (importable, exercised Phase 2)
- `flash.py` — Flash Attention wrapper (optional dep: `flash-attn`; degrades to standard MHA if unavailable)
- Export from `nn_timeline.layers.attention`

**Acceptance criteria:**
- Bahdanau: context vector shape `(batch, 1, enc_dim)` for given query + keys
- Luong: same shape contract, all three scoring modes selectable
- MHA: output shape `(batch, seq, dim)`, attention weights returnable
- MHA causal mask works correctly (no future token leakage)
- All run on MPS / CUDA / CPU

---

### T1.5 — Embeddings (`layers/embeddings/`)

Full embedding library — the most-imported surface for external packages.

**Includes:**
- `token.py` — learnable token embedding with padding mask support
- `sinusoidal.py` — fixed sinusoidal PE (Vaswani 2017), not learnable
- `learned.py` — learnable absolute PE
- `rope.py` — Rotary Positional Embedding (Phase 1 implementation, exercised Phase 2)
- `alibi.py` — ALiBi (attention with linear biases) stub
- Export from `nn_timeline.layers.embeddings`

**Acceptance criteria:**
- `SinusoidalPE(max_len=512, dim=512)(x)` adds correct PE and matches reference formula
- `RoPE` applies rotation correctly, matches reference numerics
- All embeddings handle `(batch, seq)` token input → `(batch, seq, dim)` output
- `TokenEmbedding` correctly zeroes out padding positions

---

## Tier 2 — Architectures (after T1)

### T2.1 — Migrate Transformer to `archs/tnn/`

Move and clean up existing sitMT Transformer to its new home. This is migration
+ modernization, not a rewrite.

**Includes:**
- `transformer.py` — original Vaswani Transformer (enc-dec)
- `bert.py` — encoder-only with MLM head (stub sufficient for Phase 1)
- `gpt.py` — decoder-only causal LM
- All use T1 layers (not their own reimplementations)
- Config dataclass in `configs/tnn_configs.py`

**Acceptance criteria:**
- `Transformer(config)(src, tgt)` returns `(batch, tgt_len, vocab_size)`
- `GPT(config)(x)` returns `(batch, seq, vocab_size)`
- Small config (6 layers, 8 heads, dim=512) runs on M2 with batch=32, seq=128
- No layer reimplementations — uses `nn_timeline.layers.*` throughout

---

### T2.2 — RNN family (`archs/rnn/`)

Vanilla RNN, LSTM, GRU as standalone sequence models.

**Includes:**
- `rnn.py` — stacked Vanilla RNN encoder/decoder
- `lstm.py` — stacked LSTM (encoder and decoder-only variants)
- `gru.py` — stacked GRU
- All composed from T1.1 recurrent cells
- Config dataclass in `configs/rnn_configs.py`

**Acceptance criteria:**
- `LSTM(config)(x)` returns `(output, (h_n, c_n))`
- Bidirectional support via config flag
- Runs on MPS / CUDA / CPU
- Parameter count sanity-checked against a known reference

---

### T2.3 — Seq2SeqRNN (`archs/rnn/seq2seq.py`)

RNN encoder-decoder with Bahdanau or Luong attention. This is the arch that
validates the full MT pipeline in the Phase 1 milestone.

**Includes:**
- Encoder: stacked LSTM/GRU reading source
- Decoder: stacked LSTM/GRU with attention over encoder outputs
- Attention selectable: Bahdanau | Luong (from `layers/attention/`)
- `forward(src, tgt)` → `(batch, tgt_len, vocab_size)`
- Same interface as `Transformer(config)(src, tgt)` — unified task contract

**Acceptance criteria:**
- `Seq2SeqRNN(config)(src, tgt)` output shape correct
- Attention weights returnable for inspection
- Runs on single M* without OOM for small config (hidden=256, layers=2)
- `Seq2SeqRNN` and `Transformer` are interchangeable in the `Trainer` — same call signature

---

## Tier 3 — Data + Device (parallel with Tier 2)

### T3.1 — Device abstraction (`utils/device.py`)

Central device routing. No CUDA assumptions anywhere in the codebase.

**Includes:**
- `get_device()` → `"cuda"` | `"mps"` | `"cpu"` in priority order
- `to_device(tensor_or_module)` — convenience wrapper
- Respects env var override: `NN_TIMELINE_DEVICE=cpu`

**Acceptance criteria:**
- Returns MPS on Apple Silicon Mac with no CUDA
- Returns CUDA on CUDA-capable machine
- Falls back to CPU cleanly
- All existing T1/T2 code routes through this — no hardcoded `.cuda()` calls

---

### T3.2 — Dictionary + BPE tokenizer (`data/`)

Vocabulary management and minimal subword tokenization.

**Includes:**
- `dictionary.py` — token↔id mapping, special tokens (PAD, EOS, BOS, UNK), frequency-based vocab building (migrate from sitMT)
- `bpe.py` — minimal BPE: train from raw text, encode/decode, save/load vocab
  - Self-contained — no dependency on sentencepiece for BPE itself; sentencepiece as an optional alternative backend

**Acceptance criteria:**
- BPE trains on a 100k sentence corpus in < 2 min on M2
- Encode → decode roundtrip: `decode(encode(s)) == s` for in-vocab tokens
- `Dictionary.save()` / `Dictionary.load()` roundtrip lossless
- Integrates with `LanguagePairDataset`

---

### T3.3 — Dataset pipeline (`data/datasets/` + `data/iterators.py`)

Parallel corpus loading, binarization, batching.

**Includes:**
- `indexed.py` — memory-mapped binary dataset (migrate from sitMT)
- `langpair.py` — `LanguagePairDataset`: source + target loading, padding, token prepend/append
- `lm_dataset.py` — `LMDataset`: sliding window over a flat token sequence
- `iterators.py` — `DataIterator`: batch by max tokens OR batch size, sorted for efficiency

**Acceptance criteria:**
- `LanguagePairDataset` loads IWSLT14 de-en without error
- `DataIterator` with `max_tokens=4096` produces correctly padded batches
- Memory-mapped dataset reads faster than plain text for > 100k sentences
- LMDataset produces `(input, target)` pairs with correct causal offset

---

## Tier 4 — Training (after T2 + T3)

### T4.1 — Trainer with structured returns (`train/trainer.py`)

Arch-agnostic training loop. Works with any model that respects the unified
task interface.

**Includes:**
- `Trainer(model, task, config)` — main training orchestrator
- Training loop: forward → loss → backward → step → log
- Validation loop with metric aggregation
- Early stopping support
- **Structured return**: `TrainResult(epoch, loss, metrics: dict, checkpoint_path: str)` — JSON-serializable
- Gradient accumulation (effective large batch on single GPU)
- bf16 where supported (MPS / CUDA), fp32 fallback

**Acceptance criteria:**
- `result = trainer.train()` returns `TrainResult`
- `result.metrics` is a plain dict (no tensor values — all Python scalars)
- Same `Trainer` works with `Seq2SeqRNN` and `Transformer` without modification
- Trains LSTM Seq2Seq on IWSLT14 toy subset without crashing on M2

---

### T4.2 — Optimizer + schedulers (`train/optimizer.py`, `train/scheduler.py`)

**Includes:**
- `optimizer.py` — AdamW (primary), SGD (for RNN comparison experiments)
- `scheduler.py` — InverseSquareRootSchedule (original Transformer warmup), CosineDecayWithWarmup (Phase 2 focus but implement now)
- Both schedulers configurable via `configs/`

**Acceptance criteria:**
- InvSqrtRoot LR matches formula: `lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})`
- Scheduler state saved and restored correctly through checkpoint
- `optimizer.zero_grad()` + `scaler.step()` work under bf16

---

### T4.3 — Checkpoint + HF Hub loading (`train/checkpoint.py`)

**Includes:**
- `save(model, optimizer, scheduler, step, path)` — full checkpoint
- `load(path)` — restore full state
- `load_from_hub(model_id)` — download checkpoint from HF Hub, return ready model
  - Uses `huggingface_hub` as optional dependency (warn clearly if not installed)

**Acceptance criteria:**
- Save → load roundtrip: model outputs identical before and after
- `load_from_hub("nn-timeline/transformer-mt-en-de-small")` fetches and loads without error
- Checkpoint includes: model state, optimizer state, scheduler state, config, step, metrics

---

## Tier 5 — Generate + Metrics (after T4)

### T5.1 — SequenceGenerator (`generate/generator.py`)

**Includes:**
- Beam search with configurable beam size, length penalty, min/max length
- Greedy decoding (beam=1)
- Temperature + top-p sampling for LM
- `kv_cache.py` — KV cache for efficient autoregressive decoding (stub in Phase 1, exercised Phase 2)
- **Structured return**: `GenerationResult(tokens: list, scores: list, attention: Tensor | None)`

**Acceptance criteria:**
- Beam search produces correct output on a known-good checkpoint
- `result.tokens` is a plain Python list (not tensor) — JSON-serializable
- `result.attention` is a tensor or None — not forced
- Streamed generation: `generator.generate_stream(x)` yields one token at a time

---

### T5.2 — Metrics (`metrics/`)

**Includes:**
- `bleu.py` — corpus-level BLEU (sacrebleu-compatible, no sacrebleu dep required)
- `perplexity.py` — per-token NLL → PPL
- `meters.py` — AverageMeter, TimeMeter, StopwatchMeter (migrate from sitMT)
- `chrf.py` — chrF score (implement from scratch, ~50 lines)
- `comet.py` — thin wrapper; raises `ImportError` with install instructions if `comet-ml` absent

**Acceptance criteria:**
- BLEU output matches sacrebleu on IWSLT14 de-en test set (within 0.1)
- chrF matches reference implementation on same set
- All metrics return Python scalars, not tensors
- `metrics.BLEU()(hypotheses, references)` clean one-liner API

---

## Tier 6 — Tests + Milestone (after T5)

### T6.1 — Layer unit tests (`tests/test_layers.py`)

**Includes:**
- Shape tests for every layer (given input → expected output shape)
- Numerical correctness for Bahdanau, MHA, SinusoidalPE, RoPE (vs. reference)
- Device tests: each layer runs on CPU, and MPS if available
- Gradient flow: `loss.backward()` without NaN on each layer

**Acceptance criteria:**
- All tests pass on CPU
- All tests pass on MPS (tested on M2)
- Zero NaN gradients in any forward+backward pass

---

### T6.2 — Arch unit tests (`tests/test_archs.py`)

**Includes:**
- Forward pass shape test for: `Seq2SeqRNN`, `Transformer`, `GPT`
- Interchangeability test: `Trainer` runs with both `Seq2SeqRNN` and `Transformer` on same dummy data
- Parameter count sanity: Transformer-base ~65M, small config ~10M

**Acceptance criteria:**
- All three archs produce correct output shapes
- Trainer interchangeability test passes
- No OOM on M2 with small config (dim=256, layers=2, heads=4)

---

### T6.3 — Integration milestone: Seq2Seq trains end-to-end ✓

**This is the Phase 1 validation milestone. All prior tickets must be done.**

**Setup:**
- Dataset: IWSLT14 de-en, 160k sentence pairs (or a 10k toy subset for CI)
- Model: `Seq2SeqRNN` (LSTM, hidden=256, layers=2, Bahdanau attention)
- Training: 10 epochs, AdamW, InvSqrtRoot schedule

**Acceptance criteria:**
- Trains to completion without crash on M2 (MPS) in < 60 min for 10 epochs
- Validation BLEU > 5 on IWSLT14 de-en dev set (toy baseline, not SOTA)
- `TrainResult` returned with correct structure
- Checkpoint saved and reloadable
- `SequenceGenerator` produces readable German→English output from the checkpoint

---

## Tier 7 — Notes + Web (after T6.3)

### T7.1 — `notes/00_foundations/`

Notes-only section. No arch code — pure math and intuition.

**Notebooks:**
- `backprop.ipynb` — chain rule, computational graph, manual gradient
- `mlp.ipynb` — perceptron → MLP → universal approximation
- `gradient_flow.ipynb` — vanishing/exploding gradients, why RNNs are hard

**Acceptance criteria:**
- Each notebook runs end-to-end with only `pip install nn-timeline` (+ matplotlib)
- No `from nn_timeline.*` imports (these are pre-arch notes)
- Code references header: "No arch code — see notes/01_rnn for first implementation"
- Math renders correctly in Jupyter and Quarto

---

### T7.2 — `notes/01_rnn/`

**Notebooks:**
- `rnn_intuition.ipynb` — sequence processing, hidden state, unrolling
- `lstm_gru.ipynb` — gating mechanisms, cell state, why they work
- `seq2seq.ipynb` — encoder-decoder, teacher forcing, exposure bias
- `attention_bahdanau.ipynb` — alignment, context vector, visualization

**Acceptance criteria:**
- All notebooks import from `nn_timeline.archs.rnn` and `nn_timeline.layers.*`
- Attention notebook includes a working attention heatmap (matplotlib)
- Code references header present with correct module paths
- Runs on Colab without modification (test once)

---

### T7.3 — `notes/02_transformer/`

**Notebooks:**
- `self_attention.ipynb` — QKV, scaled dot-product, multi-head
- `transformer_walkthrough.ipynb` — full forward pass, layer by layer
- `bert_gpt.ipynb` — masked LM vs. causal LM, pre-training objectives
- `positional_encoding.ipynb` — sinusoidal PE derivation + visualization

**Acceptance criteria:**
- All import from `nn_timeline.archs.tnn` and `nn_timeline.layers.*`
- PE notebook plots sinusoidal patterns correctly
- Self-attention notebook shows attention patterns on toy input

---

### T7.4 — Quarto setup + GitHub Pages

**Includes:**
- `notes/_quarto.yml` — site config, navigation mirroring timeline order
- GitHub Actions workflow: push to `main` → `quarto render` → deploy to `gh-pages`
- Static Mermaid timeline diagram (from Appendix B) embedded in site landing page
- Link from site → HF Hub for checkpoints, link → PyPI for install

**Acceptance criteria:**
- `quarto render notes/` completes without error locally
- GitHub Pages site live and navigable
- All notebooks render correctly as HTML (math, code, plots)
- Timeline diagram visible on landing page

---

## Tier 8 — HuggingFace Hub (after T6.3 + T7)

### T8.1 — First checkpoint: Transformer MT

Train and publish the first pre-trained checkpoint.

**Includes:**
- Model: `Transformer` (small config: 6 layers, 8 heads, dim=512)
- Task: MT en→de (IWSLT14) or en→am (low-resource)
- Upload to `nn-timeline/transformer-mt-en-de-small` (or `-en-am-small`)
- `models/README.md` — index of available checkpoints
- Model card: arch, training data, config, BLEU score, how to load

**Acceptance criteria:**
- `load_from_hub("nn-timeline/transformer-mt-en-de-small")` returns working model
- BLEU ≥ 15 on IWSLT14 de-en test set (small model baseline)
- Model card complete
- Demo-ready: produces readable translations in < 1s per sentence on M2

---

## Summary Table

| Ticket | Tier | Depends on | Effort est. |
|---|---|---|---|
| T0.1 Repo restructure | 0 | — | S |
| T0.2 pyproject.toml | 0 | — | S |
| T0.3 timeline_registry | 0 | — | S |
| T1.1 Recurrent cells | 1 | T0 | S |
| T1.2 Norm layers | 1 | T0 | S |
| T1.3 FFN layers | 1 | T0 | S |
| T1.4 Attention layers | 1 | T0 | M |
| T1.5 Embeddings | 1 | T0 | M |
| T2.1 Transformer (migrate) | 2 | T1 | M |
| T2.2 RNN family | 2 | T1.1 | M |
| T2.3 Seq2SeqRNN | 2 | T1.1, T1.4 | M |
| T3.1 Device abstraction | 3 | T0 | S |
| T3.2 Dictionary + BPE | 3 | T0 | M |
| T3.3 Dataset pipeline | 3 | T3.2 | M |
| T4.1 Trainer | 4 | T2, T3 | L |
| T4.2 Optimizer + schedulers | 4 | T0 | S |
| T4.3 Checkpoint + HF load | 4 | T4.1 | S |
| T5.1 SequenceGenerator | 5 | T4 | M |
| T5.2 Metrics | 5 | T0 | M |
| T6.1 Layer unit tests | 6 | T1 | M |
| T6.2 Arch unit tests | 6 | T2 | S |
| **T6.3 Integration milestone** | **6** | **T4, T5** | **L** |
| T7.1 Notes: foundations | 7 | T6.3 | M |
| T7.2 Notes: RNN | 7 | T6.3 | M |
| T7.3 Notes: Transformer | 7 | T6.3 | M |
| T7.4 Quarto + GitHub Pages | 7 | T7.1–3 | S |
| T8.1 HF Hub checkpoint | 8 | T6.3, T7.4 | M |

**Effort: S = hours, M = 1–2 days, L = 3–5 days**

---

*Created: 2026-04-23 | Phase 1 target: nn_timeline v1.0.0*
