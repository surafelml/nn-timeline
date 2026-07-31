# nn_timeline.supp

Implementation surface for the notes' supplementary tracks. Kept separate
from `nn_timeline.layers` / `nn_timeline.archs` (the core RNN → Transformer
2017 lineage) so the core stays minimal per the SSOT plan. See
`.tracker/ssot_plan.md` ("On scope and pace") for the decision record.

Each subdir mirrors a Quarto book part under `notes/_quarto.yml`. Status is
tracked per module, not per subdir — a subdir may be partially implemented.

| Subdir | Notes part | Planned modules |
|---|---|---|
| `training/` | Supplementary (Training Techniques) | `modern_activations.py` (GELU, SiLU, SwiGLU), `modern_optimizers.py` (Adam, AdamW), `normalization.py` (BatchNorm, GroupNorm), `regularization.py` (Dropout) |
| `vision/` | Supplementary (Vision & Generative) | `cnn_blocks.py` (LeNet/VGG/ResNet blocks), `self_supervised.py` (SimCLR, DINO, MAE), `generative.py` (VAE, GAN, minimal diffusion) |
| `classical_ml/` | Supplementary (Classical ML) | `linear_probabilistic.py`, `svm.py`, `trees_ensembles.py`, `unsupervised.py` (k-means, PCA) |
| `transformer_efficiency/` | Supplementary (Transformer) | `kv_cache.py`, `efficient_attention.py` (sparse, linear variants) |
| `rl_agents/` | Supplementary (RL & Agents) | `deep_rl.py` (DQN, PPO), `agents.py` (ReAct-style loop, tool-call dispatch) |
| `future/` | Future Outlook | `world_models.py` (JEPA-style predictor), `roads_not_taken.py` (Capsule routing, NTM memory, Neural ODE) |

**Status:** scaffold only — directories and `__init__.py` created 2026-07-18.
No modules implemented yet. Each module follows the same cycle as core
tickets: tests derived from the note's formalization first (red), then
implementation (green), then a Science/Engineering audit pass, per A2
before being marked done.

**Note:** `layers/norm/{layer_norm,rms_norm}.py` and
`layers/ffn/swiglu.py` already exist in **core** (T1.2, T1.3) — modern
activations/optimizers/normalization here cover only what core does not
(BatchNorm, GroupNorm, Adam/AdamW, GELU/SiLU as standalone functions).
