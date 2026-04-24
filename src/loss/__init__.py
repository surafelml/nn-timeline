"""
Module to handle scripts - why do we need all these imports here ?
- loss
- criterion
- optimizers
- lr schedulers
- ?
"""

# https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py


# optimizer
import importlib
import os

from src.utils import registry
from .optimizer_base import OptimizerBase #, LegacyOptimizerBase
from omegaconf import DictConfig

# __all__ = [
#     "AMPOptimizer",
#     "FairseqOptimizer",
#     "FP16Optimizer",
#     "MemoryEfficientFP16Optimizer",
#     "shard_",
# ]

(
    _build_optimizer,
    register_optimizer,
    OPTIMIZER_REGISTRY,
    OPTIMIZER_DATACLASS_REGISTRY,
) = registry.setup_registry("--optimizer", base_class=OptimizerBase, required=True)

def build_optimizer(cfg: DictConfig, params, *extra_args, **extra_kwargs):
    if all(isinstance(p, dict) for p in params):
        params = [t for p in params for t in p.values()]
    params = list(filter(lambda p: p.requires_grad, params))
    return _build_optimizer(cfg, params, *extra_args, **extra_kwargs)

# FIXME:
# automatically import any Python files in the optim/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq.optim." + file_name)


# lr scheulder
from .scheduler_base import SchedulerBase

(
    build_lr_scheduler_,
    register_lr_scheduler,
    LR_SCHEDULER_REGISTRY,
    LR_SCHEDULER_DATACLASS_REGISTRY,
) = registry.setup_registry(
    "--lr-scheduler", base_class=SchedulerBase, default="fixed"
)

def build_lr_scheduler(cfg: DictConfig, optimizer):
    return build_lr_scheduler_(cfg, optimizer)


# # automatically import any Python files in the optim/lr_scheduler/ directory
# for file in sorted(os.listdir(os.path.dirname(__file__))):
#     if file.endswith(".py") and not file.startswith("_"):
#         file_name = file[: file.find(".py")]
#         importlib.import_module("fairseq.optim.lr_scheduler." + file_name)
#

# criterion
from .criterion_base import CriterionBase
(
    build_criterion_,
    register_criterion,
    CRITERION_REGISTRY,
    CRITERION_DATACLASS_REGISTRY,
) = registry.setup_registry(
    "--criterion", base_class=CriterionBase, default="cross_entropy"
)

def build_criterion(cfg: DictConfig, task):
    return build_criterion_(cfg, task)

# # automatically import any Python files in the criterions/ directory
# for file in sorted(os.listdir(os.path.dirname(__file__))):
#     if file.endswith(".py") and not file.startswith("_"):
#         file_name = file[: file.find(".py")]
#         importlib.import_module("fairseq.criterions." + file_name)
#

