REGISTRY = {

    # ── RNN Family ────────────────────────────────────────────────────────────

    "rnn": {
        "family":     "rnn",
        "year":       1986,
        "paper":      "Rumelhart et al., 1986",
        "innovation": "Backpropagation through time; sequential hidden state",
        "note":       "notes/01_rnn/01_rnn_bptt.qmd",
        "hub_id":     None,
        "toy_config": {"input_size": 32, "hidden_size": 64, "layers": 1},
    },
    "lstm": {
        "family":     "rnn",
        "year":       1997,
        "paper":      "Hochreiter & Schmidhuber, 1997",
        "innovation": "Gating mechanisms (forget/input/output) solve vanishing gradient",
        "note":       "notes/01_rnn/02_lstm_gru.qmd",
        "hub_id":     None,
        "toy_config": {"input_size": 32, "hidden_size": 64, "layers": 1},
    },
    "gru": {
        "family":     "rnn",
        "year":       2014,
        "paper":      "Cho et al., 2014",
        "innovation": "Simplified gating — reset and update gates only",
        "note":       "notes/01_rnn/02_lstm_gru.qmd",
        "hub_id":     None,
        "toy_config": {"input_size": 32, "hidden_size": 64, "layers": 1},
    },
    "seq2seq_rnn": {
        "family":     "rnn",
        "year":       2014,
        "paper":      "Sutskever et al., 2014",
        "innovation": "Encoder-decoder architecture for variable-length sequence transduction",
        "note":       "notes/01_rnn/03_seq2seq.qmd",
        "hub_id":     None,
        "toy_config": {"embed_dim": 32, "hidden_dim": 64, "layers": 1, "vocab": 50},
    },
    "bahdanau_attention": {
        "family":     "rnn",
        "year":       2015,
        "paper":      "Bahdanau et al., 2015",
        "innovation": "Additive attention — decoder learns to align to encoder states",
        "note":       "notes/01_rnn/04_attention_bahdanau.qmd",
        "hub_id":     None,
        "toy_config": {"embed_dim": 32, "hidden_dim": 64, "attn_dim": 32},
    },
    "luong_attention": {
        "family":     "rnn",
        "year":       2015,
        "paper":      "Luong et al., 2015",
        "innovation": "Multiplicative attention — dot, general, concat score functions",
        "note":       "notes/01_rnn/05_attention_luong.qmd",
        "hub_id":     None,
        "toy_config": {"embed_dim": 32, "hidden_dim": 64},
    },

    # ── Transformer (TNN) Family ────────────────────────────────────────────────────

    "transformer": {
        "family":     "tnn",
        "year":       2017,
        "paper":      "Vaswani et al., 2017",
        "innovation": "Self-attention replaces recurrence entirely; parallelisable",
        "note":       "notes/02_transformer/02_transformer.qmd",
        "hub_id":     None,
        "toy_config": {"embed_dim": 32, "num_heads": 2, "layers": 2, "vocab": 50},
    },
    "bert": {
        "family":     "tnn",
        "year":       2018,
        "paper":      "Devlin et al., 2018",
        "innovation": "Bidirectional encoder; masked language modelling pre-training",
        "note":       "notes/02_transformer/03_bert_gpt.qmd",
        "hub_id":     None,
        "toy_config": {"embed_dim": 32, "num_heads": 2, "layers": 2, "vocab": 50},
    },
    "gpt": {
        "family":     "tnn",
        "year":       2018,
        "paper":      "Radford et al., 2018",
        "innovation": "Causal decoder-only; autoregressive language modelling",
        "note":       "notes/02_transformer/03_bert_gpt.qmd",
        "hub_id":     None,
        "toy_config": {"embed_dim": 32, "num_heads": 2, "layers": 2, "vocab": 50},
    },
    "kv_cache": {
        "family":     "tnn",
        "year":       2018,
        "paper":      "Radford et al., 2018 (implicit in GPT inference)",
        "innovation": "Cache K/V tensors during generation — O(n²) → O(n) per step",
        "note":       "notes/02_transformer/05_kv_cache.qmd",
        "hub_id":     None,
        "toy_config": {"embed_dim": 32, "num_heads": 2, "layers": 2},
    },
}


def get(arch: str) -> dict:
    """Return registry entry; raises KeyError if arch not found."""
    return REGISTRY[arch]


def families() -> list:
    """Return sorted list of unique family names."""
    return sorted({v["family"] for v in REGISTRY.values()})


def by_family(family: str) -> dict:
    """Return all entries for a given family."""
    return {k: v for k, v in REGISTRY.items() if v["family"] == family}
