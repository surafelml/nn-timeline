"""
Utility for pre/post-processing steps
"""

import re

SPACE_NORMALIZER = re.compile(r"\s+")

def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
    #from fairseq import tokenizer

    # Tokens are strings here
    hypo_tokens = tokenize_line(hypo_str)
    # TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
    src_tokens = tokenize_line(src_str) + ["<eos>"]
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            src_token = src_tokens[alignment[i]]
            # Either take the corresponding value in the aligned dictionary or just copy the original value.
            hypo_tokens[i] = align_dict.get(src_token, src_token)
    return " ".join(hypo_tokens)




# from utils - used after prediction
def post_process_prediction(
    hypo_tokens,
    src_str,
    #alignment,
    #align_dict,
    tgt_dict,
    remove_bpe=None,
    extra_symbols_to_ignore=None,
):
    hypo_str = tgt_dict.string(
        hypo_tokens, remove_bpe, extra_symbols_to_ignore=extra_symbols_to_ignore
    )
    # note: no alignment
    # if align_dict is not None:
    #     hypo_str = replace_unk(
    #         hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string()
    #     )
    # if align_dict is not None or remove_bpe is not None:
    #     # Convert back to tokens for evaluating with unk replacement or without BPE
    #     # Note that the dictionary can be modified inside the method.
    #     hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
    return hypo_tokens, hypo_str #, alignment