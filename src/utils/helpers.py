"""
Helper function/utilities without complex ops
"""

import torch
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
from src.layers.shared import gelu
import torch.nn.functional as F
import warnings # ?


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    #from fairseq.modules import gelu, gelu_accurate

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    # elif activation == "gelu_fast":
    #     deprecation_warning(
    #         "--activation-fn=gelu_fast has been renamed to gelu_accurate"
    #     )
    #     return gelu_accurate
    # elif activation == "gelu_accurate":
    #     return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


# from fairseq/utils - for trainer
def has_parameters(module):
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


# FROM FAIRSEQ UTILS
import os
MANIFOLD_PATH_SEP = "|"
def split_paths(paths: str, separator=os.pathsep) -> List[str]:
    return (
        paths.split(separator) if "://" not in paths else paths.split(MANIFOLD_PATH_SEP)
    )


# note: not sure why this is important
def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            max_positions, arg = _match_types(max_positions, arg)
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(map(nullsafe_min, zip(max_positions, arg)))

    return max_positions





# the following func should go to data utils
def item(tensor):
    # tpu-comment: making this a no-op for xla devices.
    if torch.is_tensor(tensor) and tensor.device.type == "xla":
        return tensor.detach()
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


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

import re
SPACE_NORMALIZER = re.compile(r"\s+")
def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()





# this also should go to data utils
import os
import typing as tp


def _safe_readline(fd) -> str:
    pos = fd.tell()
    while True:
        try:
            return fd.readline()
        except UnicodeDecodeError:
            pos -= 1
            fd.seek(pos)  # search where this character begins


def find_offsets(filename: str, num_chunks: int) -> tp.List[int]:
    """
    given a file and a number of chuncks, find the offsets in the file
    to be able to chunk around full lines.
    """
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            _safe_readline(f)
            offsets[i] = f.tell()
        offsets[-1] = size
        return offsets


class ChunkLineIterator:
    """
    Iterator to properly iterate over lines of a file chunck.
    """

    def __init__(self, fd, start_offset: int, end_offset: int):
        self._fd = fd
        self._start_offset = start_offset
        self._end_offset = end_offset

    def __iter__(self) -> tp.Iterable[str]:
        self._fd.seek(self._start_offset)
        # next(f) breaks f.tell(), hence readline() must be used
        line = _safe_readline(self._fd)
        while line:
            pos = self._fd.tell()
            # f.tell() does not always give the byte position in the file
            # sometimes it skips to a very large number
            # it is unlikely that through a normal read we go from
            # end bytes to end + 2**32 bytes (4 GB) and this makes it unlikely
            # that the procedure breaks by the undeterministic behavior of
            # f.tell()
            if (
                self._end_offset > 0
                and pos > self._end_offset
                and pos < self._end_offset + 2 ** 32
            ):
                break
            yield line
            line = self._fd.readline()


class Chunker:
    """
    contextmanager to read a chunck of a file line by line.
    """

    def __init__(self, path: str, start_offset: int, end_offset: int):
        self.path = path
        self.start_offset = start_offset
        self.end_offset = end_offset

    def __enter__(self) -> ChunkLineIterator:
        self.fd = open(self.path, "r", encoding="utf-8")
        return ChunkLineIterator(self.fd, self.start_offset, self.end_offset)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.fd.close()



# Generator/Inference helpers
def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)

def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)


# note: move the sample to ....
def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32.
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]



# optimizer - loss related func/s  - move to './loss' module ?
# v1.0 not using "multi_tensor_total_norm"
try:
    from amp_C import multi_tensor_l2norm
    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False

@torch.no_grad()
def clip_grad_norm_(params, max_norm, aggregate_norm_fn=None) -> torch.Tensor:
    def grad_exists(p):
        return p is not None and getattr(p, "grad", None) is not None

    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [
        p.grad.detach() for p in params if grad_exists(p) and not hasattr(p, "expert")
    ]
    expert_grads = [
        p.grad.detach() for p in params if grad_exists(p) and hasattr(p, "expert")
    ]

    if len(grads) == 0:
        if len(params) > 0:
            return params[0].new_tensor(0.0)
        else:
            return torch.tensor(0.0)

    if len(grads) == 1:
        total_norm = torch.norm(grads[0], p=2, dtype=torch.float32)
    else:
        # not in v1.0
        # if multi_tensor_l2norm_available:
        #     total_norm = multi_tensor_total_norm(grads)
        # else:
        if torch.cuda.is_available():
            warnings.warn(
                "amp_C fused kernels unavailable, disabling multi_tensor_l2norm; "
                "you may get better performance by installing NVIDIA's apex library"
            )
            device = torch.cuda.current_device()
        elif grads[0].device.type == "xla":
            device = grads[0].device
        else:
            device = torch.device("cpu")
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(g, p=2, dtype=torch.float32).to(device) for g in grads]
            )
        )

    if aggregate_norm_fn is not None:
        total_norm = aggregate_norm_fn(total_norm)

    if max_norm > 0:
        max_norm = float(max_norm)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp_(max=1)
        for g in grads + expert_grads:
            g.mul_(clip_coef)
    return total_norm


# new: torch seeding
# move to some_utils (seeding.py? as a generic script)
def seeding(seed):
    assert isinstance(seed, int)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



# from utils.py - move to dir_io.py ?
# note: no need in v1.0 as we don't load modules (like discriminative_rerank) from # user module.
import sys
import importlib
def import_user_module(args):
    module_path = getattr(args, "user_dir", None)  # == args.user_dir
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        if not os.path.exists(module_path) and not os.path.isfile(
            os.path.dirname(module_path)
        ):
            fairseq_rel_path = os.path.join(os.path.dirname(__file__), args.user_dir)
            if os.path.exists(fairseq_rel_path):
                module_path = fairseq_rel_path
            else:
                fairseq_rel_path = os.path.join(
                    os.path.dirname(__file__), "..", args.user_dir
                )
                if os.path.exists(fairseq_rel_path):
                    module_path = fairseq_rel_path
                else:
                    raise FileNotFoundError(module_path)

        # ensure that user modules are only imported once
        import_user_module.memo = getattr(import_user_module, "memo", set())
        if module_path not in import_user_module.memo:
            import_user_module.memo.add(module_path)

            module_parent, module_name = os.path.split(module_path)
            if module_name not in sys.modules:
                sys.path.insert(0, module_parent)
                importlib.import_module(module_name)

                tasks_path = os.path.join(module_path, "tasks")
                if os.path.exists(tasks_path):
                    from fairseq.tasks import import_tasks
                    #from sitMT.tasks import

                    import_tasks(tasks_path, f"{module_name}.tasks")

                models_path = os.path.join(module_path, "models")
                if os.path.exists(models_path):
                    #from fairseq.models import import_models
                    from src.models import import_models

                    import_models(models_path, f"{module_name}.models")
            else:
                raise ImportError(
                    "Failed to import --user-dir={} because the corresponding module name "
                    "({}) is not globally unique. Please rename the directory to "
                    "something unique and try again.".format(module_path, module_name)
                )



# from fairseq/dataclass/utils.py  - used for config
import ast
def eval_str_list(x, x_type=float):
    if x is None:
        return None
    if isinstance(x, str):
        if len(x) == 0:
            return []
        x = ast.literal_eval(x)
    try:
        return list(map(x_type, x))
    except TypeError:
        return [x_type(x)]


def interpret_dc_type(field_type):
    if isinstance(field_type, str):
        raise RuntimeError("field should be a type")

    if field_type == Any:
        return str

    typestring = str(field_type)
    if re.match(
        r"(typing.|^)Union\[(.*), NoneType\]$", typestring
    ) or typestring.startswith("typing.Optional"):
        return field_type.__args__[0]
    return field_type





# from fairseq/nan_detector.py for use in Training, - check need, simplify, move to proper script
import logging
import torch
logger = logging.getLogger(__name__)

class NanDetector:
    """
    Detects the first NaN or Inf in forward and/or backward pass and logs, together with the module name
    """

    def __init__(self, model, forward=True, backward=True):
        self.bhooks = []
        self.fhooks = []
        self.forward = forward
        self.backward = backward
        self.named_parameters = list(model.named_parameters())
        self.reset()

        for name, mod in model.named_modules():
            mod.__module_name = name
            self.add_hooks(mod)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Dump out all model gnorms to enable better debugging
        norm = {}
        gradients = {}
        for name, param in self.named_parameters:
            if param.grad is not None:
                grad_norm = torch.norm(param.grad.data, p=2, dtype=torch.float32)
                norm[name] = grad_norm.item()
                if torch.isnan(grad_norm).any() or torch.isinf(grad_norm).any():
                    gradients[name] = param.grad.data
        if len(gradients) > 0:
            logger.info("Detected nan/inf grad norm, dumping norms...")
            logger.info(f"norms: {norm}")
            logger.info(f"gradients: {gradients}")

        self.close()

    def add_hooks(self, module):
        if self.forward:
            self.fhooks.append(module.register_forward_hook(self.fhook_fn))
        if self.backward:
            self.bhooks.append(module.register_backward_hook(self.bhook_fn))

    def reset(self):
        self.has_printed_f = False
        self.has_printed_b = False

    def _detect(self, tensor, name, backward):
        err = None
        if (
            torch.is_floating_point(tensor)
            # single value tensors (like the loss) will not provide much info
            and tensor.numel() >= 2
        ):
            with torch.no_grad():
                if torch.isnan(tensor).any():
                    err = "NaN"
                elif torch.isinf(tensor).any():
                    err = "Inf"
        if err is not None:
            err = f"{err} detected in output of {name}, shape: {tensor.shape}, {'backward' if backward else 'forward'}"
        return err

    def _apply(self, module, inp, x, backward):
        if torch.is_tensor(x):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            err = self._detect(x, module.__module_name, backward)
            if err is not None:
                if torch.is_tensor(inp) and not backward:
                    err += (
                        f" input max: {inp.max().item()}, input min: {inp.min().item()}"
                    )

                has_printed_attr = "has_printed_b" if backward else "has_printed_f"
                logger.warning(err)
                setattr(self, has_printed_attr, True)
        elif isinstance(x, dict):
            for v in x.values():
                self._apply(module, inp, v, backward)
        elif isinstance(x, list) or isinstance(x, tuple):
            for v in x:
                self._apply(module, inp, v, backward)

    def fhook_fn(self, module, inp, output):
        if not self.has_printed_f:
            self._apply(module, inp, output, backward=False)

    def bhook_fn(self, module, inp, output):
        if not self.has_printed_b:
            self._apply(module, inp, output, backward=True)

    def close(self):
        for hook in self.fhooks + self.bhooks:
            hook.remove()

