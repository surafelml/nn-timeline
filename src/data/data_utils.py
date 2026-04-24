"""
Only keep utils and move the rest to organize: dataset_batching, etc
"""


try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import contextlib
import itertools
import logging
import re
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

#from fairseq.file_io import PathManager
from src.utils.file_io import PathManager
#from fairseq import utils
from src.utils import helpers
import os

logger = logging.getLogger(__name__)


# from fairseq/data/data_utils
def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in PathManager.ls(path):
        parts = filename.split(".")
        if len(parts) >= 3 and len(parts[1].split("-")) == 2:
            return parts[1].split("-")
    return src, dst


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


# def index_file_path(prefix_path):
#     return prefix_path + ".idx"
#
# def data_file_path(prefix_path):
#     return prefix_path + ".bin"
#
# def get_indexed_dataset_to_local(path) -> str:
#     local_index_path = PathManager.get_local_path(index_file_path(path))
#     local_data_path = PathManager.get_local_path(data_file_path(path))
#
#     assert local_index_path.endswith(".idx") and local_data_path.endswith(".bin"), (
#         "PathManager.get_local_path does not return files with expected patterns: "
#         f"{local_index_path} and {local_data_path}"
#     )
#
#     local_path = local_data_path[:-4]  # stripping surfix ".bin"
#     assert local_path == local_index_path[:-4]  # stripping surfix ".idx"
#     return local_path
#
#
# def infer_dataset_impl(path):
#     if IndexedRawTextDataset.exists(path):
#         return "raw"
#     elif IndexedDataset.exists(path):
#         with open(index_file_path(path), "rb") as f:
#             magic = f.read(8)
#             if magic == IndexedDataset._HDR_MAGIC:
#                 return "cached"
#             elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
#                 return "mmap"
#             else:
#                 return None
#     elif FastaDataset.exists(path):
#         return "fasta"
#     else:
#         return None


def load_indexed_dataset(
    path, dictionary=None, dataset_impl=None, combine=False, default="cached"
):
    """A helper function for loading indexed datasets.

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~fairseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    #import fairseq.data.indexed_dataset as indexed_dataset
    from src.data import indexed_dataset
    #from fairseq.data.concat_dataset import ConcatDataset


    datasets = []
    for k in itertools.count():
        path_k = path + (str(k) if k > 0 else "")
        try:
            path_k = indexed_dataset.get_indexed_dataset_to_local(path_k)

        except Exception as e:
            if "StorageException: [404] Path not found" in str(e):
                logger.warning(f"path_k: {e} not found")
            else:
                raise e

        dataset_impl_k = dataset_impl
        if dataset_impl_k is None:
            dataset_impl_k = indexed_dataset.infer_dataset_impl(path_k)
        dataset = indexed_dataset.make_dataset(
            path_k,
            impl=dataset_impl_k or default,
            fix_lua_indexing=True,
            dictionary=dictionary,
        )
        if dataset is None:
            break
        logger.info("loaded {:,} examples from: {}".format(len(dataset), path_k))
        datasets.append(dataset)
        if not combine:
            break
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False):
    def compare_leq(a, b):
        return a <= b if not isinstance(a, tuple) else max(a) <= b

    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(
                    a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key])
                )
                for key in intersect_keys
            )
        else:
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                return all(size_fn(idx) <= b for b in max_positions)
            return all(
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )

    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    indices = np.fromiter(itr, dtype=np.int64, count=-1)
    return indices, ignored


def filter_by_size(indices, dataset, max_positions, raise_exception=False):
    """
    [deprecated] Filter indices based on their size.
    Use `FairseqDataset::filter_indices_by_size` instead.

    Args:
        indices (List[int]): ordered list of dataset indices
        dataset (FairseqDataset): fairseq dataset instance
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception if
            any elements are filtered (default: False).
    """
    warnings.warn(
        "data_utils.filter_by_size is deprecated. "
        "Use `FairseqDataset::filter_indices_by_size` instead.",
        stacklevel=2,
    )
    if isinstance(max_positions, float) or isinstance(max_positions, int):
        if hasattr(dataset, "sizes") and isinstance(dataset.sizes, np.ndarray):
            ignored = indices[dataset.sizes[indices] > max_positions].tolist()
            indices = indices[dataset.sizes[indices] <= max_positions]
        elif (
            hasattr(dataset, "sizes")
            and isinstance(dataset.sizes, list)
            and len(dataset.sizes) == 1
        ):
            ignored = indices[dataset.sizes[0][indices] > max_positions].tolist()
            indices = indices[dataset.sizes[0][indices] <= max_positions]
        else:
            indices, ignored = _filter_by_size_dynamic(
                indices, dataset.size, max_positions
            )
    else:
        indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)

    if len(ignored) > 0 and raise_exception:
        raise Exception(
            (
                "Size of sample #{} is invalid (={}) since max_positions={}, "
                "skip this example with --skip-invalid-size-inputs-valid-test"
            ).format(ignored[0], dataset.size(ignored[0]), max_positions)
        )
    if len(ignored) > 0:
        logger.warning(
            (
                "{} samples have invalid sizes and will be skipped, "
                "max_positions={}, first few sample ids={}"
            ).format(len(ignored), max_positions, ignored[:10])
        )
    return indices


def filter_paired_dataset_indices_by_size(src_sizes, tgt_sizes, indices, max_sizes):
    """Filter a list of sample indices. Remove those that are longer
        than specified in max_sizes.

    Args:
        indices (np.array): original array of sample indices
        max_sizes (int or list[int] or tuple[int]): max sample size,
            can be defined separately for src and tgt (then list or tuple)

    Returns:
        np.array: filtered sample array
        list: list of removed indices
    """
    if max_sizes is None:
        return indices, []
    if type(max_sizes) in (int, float):
        max_src_size, max_tgt_size = max_sizes, max_sizes
    else:
        max_src_size, max_tgt_size = max_sizes
    if tgt_sizes is None:
        ignored = indices[src_sizes[indices] > max_src_size]
    else:
        ignored = indices[
            (src_sizes[indices] > max_src_size) | (tgt_sizes[indices] > max_tgt_size)
        ]
    if len(ignored) > 0:
        if tgt_sizes is None:
            indices = indices[src_sizes[indices] <= max_src_size]
        else:
            indices = indices[
                (src_sizes[indices] <= max_src_size)
                & (tgt_sizes[indices] <= max_tgt_size)
            ]
    return indices, ignored.tolist()


def batch_by_size(
    indices,
    num_tokens_fn,
    num_tokens_vec=None,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
    fixed_shapes=None,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        num_tokens_vec (List[int], optional): precomputed vector of the number
            of tokens for each index in indices (to enable faster batch generation)
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be less than N or a multiple of N (default: 1).
        fixed_shapes (List[Tuple[int, int]], optional): if given, batches will
            only be created with the given shapes. *max_sentences* and
            *required_batch_size_multiple* will be ignored (default: None).
    """
    try:
        from fairseq.data.data_utils_fast import (
            batch_by_size_fn,
            batch_by_size_vec,
            batch_fixed_shapes_fast,
        )
    except ImportError:
        raise ImportError(
            "Please build Cython components with: "
            "`python setup.py build_ext --inplace`"
        )
    except ValueError:
        raise ValueError(
            "Please build (or rebuild) Cython components with `python setup.py build_ext --inplace`."
        )

    # added int() to avoid TypeError: an integer is required
    max_tokens = (
        int(max_tokens) if max_tokens is not None else -1
    )
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    if num_tokens_vec is not None and not isinstance(num_tokens_vec, np.ndarray):
        num_tokens_vec = np.fromiter(num_tokens_vec, dtype=np.int64, count=-1)

    if fixed_shapes is None:
        if num_tokens_vec is None:
            return batch_by_size_fn(
                indices,
                num_tokens_fn,
                max_tokens,
                max_sentences,
                bsz_mult,
            )
        else:
            return batch_by_size_vec(
                indices,
                num_tokens_vec,
                max_tokens,
                max_sentences,
                bsz_mult,
            )

    else:
        fixed_shapes = np.array(fixed_shapes, dtype=np.int64)
        sort_order = np.lexsort(
            [
                fixed_shapes[:, 1].argsort(),  # length
                fixed_shapes[:, 0].argsort(),  # bsz
            ]
        )
        fixed_shapes_sorted = fixed_shapes[sort_order]
        return batch_fixed_shapes_fast(indices, num_tokens_fn, fixed_shapes_sorted)


def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "silence":
        import re
        sentence = sentence.replace("<SIL>", "")
        sentence = re.sub(' +', ' ', sentence).strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol in {"subword_nmt", "@@ ", "@@"}:
        if symbol == "subword_nmt":
            symbol = "@@ "
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


def get_mem_usage():
    try:
        import psutil

        mb = 1024 * 1024
        return f"used={psutil.virtual_memory().used / mb}Mb; avail={psutil.virtual_memory().available / mb}Mb"
    except ImportError:
        return "N/A"


# lens: torch.LongTensor
# returns: torch.BoolTensor
def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


# lens: torch.LongTensor
# returns: torch.BoolTensor
def lengths_to_mask(lens):
    return ~lengths_to_padding_mask(lens)


def get_buckets(sizes, num_buckets):
    buckets = np.unique(
        np.percentile(
            sizes,
            np.linspace(0, 100, num_buckets + 1),
            interpolation='lower',
        )[1:]
    )
    return buckets


def get_bucketed_sizes(orig_sizes, buckets):
    sizes = np.copy(orig_sizes)
    assert np.min(sizes) >= 0
    start_val = -1
    for end_val in buckets:
        mask = (sizes > start_val) & (sizes <= end_val)
        sizes[mask] = end_val
        start_val = end_val
    return sizes



def _find_extra_valid_paths(dataset_path: str) -> set:
    """

    :param dataset_path:
    :return:
    """
    paths = helpers.split_paths(dataset_path)
    all_valid_paths = set()
    for sub_dir in paths:
        contents = PathManager.ls(sub_dir)
        valid_paths = [c for c in contents if re.match("valid*[0-9].*", c) is not None]
        all_valid_paths |= {os.path.basename(p) for p in valid_paths}
    # Remove .bin, .idx etc
    roots = {os.path.splitext(p)[0] for p in all_valid_paths}
    return roots


def raise_if_valid_subsets_unintentionally_ignored(train_cfg) -> None:
    """Raises if there are paths matching 'valid*[0-9].*' which are not combined or ignored."""
    if (
        train_cfg.dataset.ignore_unused_valid_subsets
        or train_cfg.dataset.combine_valid_subsets
        or train_cfg.dataset.disable_validation
        or not hasattr(train_cfg.task, "data")
    ):
        return
    other_paths = _find_extra_valid_paths(train_cfg.task.data)
    specified_subsets = train_cfg.dataset.valid_subset.split(",")
    ignored_paths = [p for p in other_paths if p not in specified_subsets]
    if ignored_paths:
        advice = "Set --combine-val to combine them or --ignore-unused-valid-subsets to ignore them."
        msg = f"Valid paths {ignored_paths} will be ignored. {advice}"
        raise ValueError(msg)








# FROM fairseq_dataset.py # TODO: move to separate dataset_batching ?
import logging
import numpy as np
import torch.utils.data
#from fairseq.data import data_utils

logger = logging.getLogger(__name__)


class EpochListening:
    """Mixin for receiving updates whenever the epoch increments."""

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        """
        Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for
        this dataset across epochs.

        This needs to return ``False`` if the sample sizes can change across
        epochs, in which case we may need to regenerate batches at each epoch.
        If your dataset relies in ``set_epoch`` then you should consider setting
        this to ``False``.
        """
        return True

    def set_epoch(self, epoch):
        """Will receive the updated epoch number at the beginning of the epoch."""
        pass


class FairseqDataset(torch.utils.data.Dataset, EpochListening):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype=np.int64)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        raise NotImplementedError

    def get_batch_shapes(self):
        """
        Return a list of valid batch shapes, for example::

            [(8, 512), (16, 256), (32, 128)]

        The first dimension of each tuple is the batch size and can be ``None``
        to automatically infer the max batch size based on ``--max-tokens``.
        The second dimension of each tuple is the max supported length as given
        by :func:`fairseq.data.FairseqDataset.num_tokens`.

        This will be used by :func:`fairseq.data.FairseqDataset.batch_by_size`
        to restrict batch shapes. This is useful on TPUs to avoid too many
        dynamic shapes (and recompilations).
        """
        return None

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        """
        Given an ordered set of indices, return batches according to
        *max_tokens*, *max_sentences* and *required_batch_size_multiple*.
        """
        #from fairseq.data import data_utils

        fixed_shapes = self.get_batch_shapes()
        if fixed_shapes is not None:

            def adjust_bsz(bsz, num_tokens):
                if bsz is None:
                    assert max_tokens is not None, "Must specify --max-tokens"
                    bsz = max_tokens // num_tokens
                if max_sentences is not None:
                    bsz = min(bsz, max_sentences)
                elif (
                    bsz >= required_batch_size_multiple
                    and bsz % required_batch_size_multiple != 0
                ):
                    bsz -= bsz % required_batch_size_multiple
                return bsz

            fixed_shapes = np.array(
                [
                    [adjust_bsz(bsz, num_tokens), num_tokens]
                    for (bsz, num_tokens) in fixed_shapes
                ]
            )

        try:
            num_tokens_vec = self.num_tokens_vec(indices).astype('int64')
        except NotImplementedError:
            num_tokens_vec = None

        return batch_by_size(  #data_utils.batch_by_size(
            indices,
            num_tokens_fn=self.num_tokens,
            num_tokens_vec=num_tokens_vec,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            fixed_shapes=fixed_shapes,
        )

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif (
                hasattr(self, "sizes")
                and isinstance(self.sizes, list)
                and len(self.sizes) == 1
            ):
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                indices, ignored = _filter_by_size_dynamic(
                    indices, self.size, max_sizes
                )
        else:
            indices, ignored = _filter_by_size_dynamic(
                indices, self.size, max_sizes
            )
        return indices, ignored

    @property
    def supports_fetch_outside_dataloader(self):
        """Whether this dataset supports fetching outside the workers of the dataloader."""
        return True


class FairseqIterableDataset(torch.utils.data.IterableDataset, EpochListening):
    """
    For datasets that need to be read sequentially, usually because the data is
    being streamed or otherwise can't be manipulated on a single machine.
    """

    def __iter__(self):
        raise NotImplementedError





# from dataclass/utils.py - move to configs ?
from src.configs.configs import FairseqDataclass
from omegaconf import DictConfig, OmegaConf, open_dict, _utils
from dataclasses import _MISSING_TYPE, MISSING, is_dataclass

def merge_with_parent(dc: FairseqDataclass, cfg: DictConfig, remove_missing=True):
    if remove_missing:

        if is_dataclass(dc):
            target_keys = set(dc.__dataclass_fields__.keys())
        else:
            target_keys = set(dc.keys())

        with open_dict(cfg):
            for k in list(cfg.keys()):
                if k not in target_keys:
                    del cfg[k]

    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg


# for registery
def merge_with_parent(dc: FairseqDataclass, cfg: DictConfig, remove_missing=True):
    if remove_missing:

        if is_dataclass(dc):
            target_keys = set(dc.__dataclass_fields__.keys())
        else:
            target_keys = set(dc.keys())

        with open_dict(cfg):
            for k in list(cfg.keys()):
                if k not in target_keys:
                    del cfg[k]

    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg


# from 'base_wrapper_dataset'
from torch.utils.data.dataloader import default_collate
#from . import FairseqDataset

class BaseWrapperDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, "collater"):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)

    @property
    def sizes(self):
        return self.dataset.sizes

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def attr(self, attr: str, index: int):
        return self.dataset.attr(attr, index)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    def get_batch_shapes(self):
        return self.dataset.get_batch_shapes()

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        return self.dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

    def filter_indices_by_size(self, indices, max_sizes):
        return self.dataset.filter_indices_by_size(indices, max_sizes)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return self.dataset.can_reuse_epoch_itr_across_epochs

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


# from 'append_token_dataset'
import numpy as np
import torch
#from . import BaseWrapperDataset

class AppendTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token
        if token is not None:
            self._sizes = np.array(dataset.sizes) + 1
        else:
            self._sizes = dataset.sizes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            item = torch.cat([item, item.new([self.token])])
        return item

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        if self.token is not None:
            n += 1
        return n

    def size(self, index):
        n = self.dataset.size(index)
        if self.token is not None:
            n += 1
        return n


# from 'concat_dataset'
import bisect
import numpy as np
from torch.utils.data.dataloader import default_collate
#from . import FairseqDataset

class ConcatDataset(FairseqDataset):
    @staticmethod
    def cumsum(sequence, sample_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, sample_ratios):
            curr_len = int(ratio * len(e))
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sample_ratios=1):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
        self.sample_ratios = sample_ratios
        self.cumulative_sizes = self.cumsum(self.datasets, sample_ratios)
        self.real_sizes = [len(d) for d in self.datasets]

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx][sample_idx]

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        if hasattr(self.datasets[0], "collater"):
            return self.datasets[0].collater(samples, **extra_args)
        else:
            return default_collate(samples, **extra_args)

    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx].size(sample_idx)

    def num_tokens(self, index: int):
        return np.max(self.size(index))

    def attr(self, attr: str, index: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        return getattr(self.datasets[dataset_idx], attr, None)

    @property
    def sizes(self):
        _dataset_sizes = []
        for ds, sr in zip(self.datasets, self.sample_ratios):
            if isinstance(ds.sizes, np.ndarray):
                _dataset_sizes.append(np.tile(ds.sizes, sr))
            else:
                # Only support underlying dataset with single size array.
                assert isinstance(ds.sizes, list)
                _dataset_sizes.append(np.tile(ds.sizes[0], sr))
        return np.concatenate(_dataset_sizes)

    @property
    def supports_prefetch(self):
        return all(d.supports_prefetch for d in self.datasets)

    def ordered_indices(self):
        """
        Returns indices sorted by length. So less padding is needed.
        """
        if isinstance(self.sizes, np.ndarray) and len(self.sizes.shape) > 1:
            # special handling for concatenating lang_pair_datasets
            indices = np.arange(len(self))
            sizes = self.sizes
            tgt_sizes = (
                sizes[:, 1] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else None
            )
            src_sizes = (
                sizes[:, 0] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else sizes
            )
            # sort by target length, then source length
            if tgt_sizes is not None:
                indices = indices[np.argsort(tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(src_sizes[indices], kind="mergesort")]
        else:
            return np.argsort(self.sizes)

    def prefetch(self, indices):
        frm = 0
        for to, ds in zip(self.cumulative_sizes, self.datasets):
            real_size = len(ds)
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch([(i - frm) % real_size for i in indices if frm <= i < to])
            frm = to

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return all(d.can_reuse_epoch_itr_across_epochs for d in self.datasets)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

# from data/strip_token_dataset
class StripTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, id_to_strip):
        super().__init__(dataset)
        self.id_to_strip = id_to_strip

    def __getitem__(self, index):
        item = self.dataset[index]
        while len(item) > 0 and item[-1] == self.id_to_strip:
            item = item[:-1]
        while len(item) > 0 and item[0] == self.id_to_strip:
            item = item[1:]
        return item


# from data/shorten_dataset
class TruncateDataset(BaseWrapperDataset):
    """Truncate a sequence by returning the first truncation_length tokens"""

    def __init__(self, dataset, truncation_length):
        super().__init__(dataset)
        assert truncation_length is not None
        self.truncation_length = truncation_length
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        item_len = item.size(0)
        if item_len > self.truncation_length:
            item = item[: self.truncation_length]
        return item

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.truncation_length)

    def __len__(self):
        return len(self.dataset)






# from data/preprend_token_dataset
import numpy as np
import torch
#from . import BaseWrapperDataset

class PrependTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token
        if token is not None:
            self._sizes = np.array(dataset.sizes) + 1
        else:
            self._sizes = dataset.sizes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            item = torch.cat([item.new([self.token]), item])
        return item

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        if self.token is not None:
            n += 1
        return n

    def size(self, index):
        n = self.dataset.size(index)
        if self.token is not None:
            n += 1
        return n



# from fairseq/utils
def parse_alignment(line):
    """
    Parses a single line from the alingment file.

    Args:
        line (str): String containing the alignment of the format:
            <src_idx_1>-<tgt_idx_1> <src_idx_2>-<tgt_idx_2> ..
            <src_idx_m>-<tgt_idx_m>. All indices are 0 indexed.

    Returns:
        torch.IntTensor: packed alignments of shape (2 * m).
    """
    alignments = line.strip().split()
    parsed_alignment = torch.IntTensor(2 * len(alignments))
    for idx, alignment in enumerate(alignments):
        src_idx, tgt_idx = alignment.split("-")
        parsed_alignment[2 * idx] = int(src_idx)
        parsed_alignment[2 * idx + 1] = int(tgt_idx)
    return parsed_alignment