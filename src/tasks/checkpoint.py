"""
Code to manage model/param checkpoint
"""


import ast
import collections
import contextlib
import logging
import numpy as np
import os
import re
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import torch
# from fairseq.data import data_utils
# from fairseq.dataclass.configs import CheckpointConfig
from src.configs.arguments import CheckpointConfig
# from fairseq.dataclass.utils import (
#     convert_namespace_to_omegaconf,
#     overwrite_args_by_name,
# )
# from fairseq.distributed.fully_sharded_data_parallel import FSDP, has_FSDP
# from fairseq.file_io import PathManager
from src.utils.file_io import PathManager
# from fairseq.models import FairseqDecoder, FairseqEncoder
from omegaconf import DictConfig, open_dict, OmegaConf
from src.metrics import meters
from src.utils import helpers


logger = logging.getLogger(__name__)




def checkpoint_paths(path, pattern=r"checkpoint(\d+)\.pt", keep_match=False):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = PathManager.ls(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = float(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    if keep_match:
        return [(os.path.join(path, x[1]), x[0]) for x in sorted(entries, reverse=True)]
    else:
        return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]



def save_checkpoint(cfg: CheckpointConfig, trainer, epoch_itr, val_loss):
    """
    func called for saving checkpoints ...
    :param cfg:
    :param trainer:
    :param epoch_itr:
    :param val_loss:
    :return:
    """
    #from fairseq import meters

    # only one worker should attempt to create the required dir
    if trainer.data_parallel_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if cfg.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if cfg.no_save:
        return

    trainer.consolidate_optimizer()  # TODO(SS): do we need this if no_save_optimizer_state

    # note:
    # if not trainer.should_save_checkpoint_on_current_rank:
    #     if trainer.always_call_state_dict_during_save_checkpoint:
    #         trainer.state_dict()
    #     return

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    logger.info(f"Preparing to save checkpoint for epoch {epoch} @ {updates} updates")

    def is_better(a, b):
        return a >= b if cfg.maximize_best_checkpoint_metric else a <= b

    # note: ?
    suffix = trainer.checkpoint_suffix
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}{}.pt".format(epoch, suffix)] = (
        end_of_epoch and not cfg.no_epoch_checkpoints and epoch % cfg.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}{}.pt".format(epoch, updates, suffix)] = (
        not end_of_epoch
        and cfg.save_interval_updates > 0
        and updates % cfg.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best{}.pt".format(suffix)] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(val_loss, save_checkpoint.best)
    )

    if val_loss is not None and cfg.keep_best_checkpoints > 0:
        worst_best = getattr(save_checkpoint, "best", None)
        chkpts = checkpoint_paths(
            cfg.save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*){}\.pt".format(
                cfg.best_checkpoint_metric, suffix
            ),
        )

        if len(chkpts) > 0:
            p = chkpts[-1] if cfg.maximize_best_checkpoint_metric else chkpts[0]
            worst_best = float(p.rsplit("_")[-1].replace("{}.pt".format(suffix), ""))

        # add random digits to resolve ties
        with helpers.numpy_seed(epoch, updates, val_loss):
            rand_sfx = np.random.randint(0, cfg.keep_best_checkpoints)

        checkpoint_conds[
            "checkpoint.best_{}_{:.3f}{}{}.pt".format(
                cfg.best_checkpoint_metric,
                val_loss,
                rand_sfx,
                suffix
            )
        ] = worst_best is None or is_better(val_loss, worst_best)
    checkpoint_conds[
        "checkpoint_last{}.pt".format(suffix)
    ] = not cfg.no_last_checkpoints

    extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})

    checkpoints = [
        os.path.join(cfg.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0:
        trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            if cfg.write_checkpoints_asynchronously:
                # TODO[ioPath]: Need to implement a delayed asynchronous
                # file copying/moving feature.
                logger.warning(
                    f"ioPath is not copying {checkpoints[0]} to {cp} "
                    "since async write mode is on."
                )
            else:
                assert PathManager.copy(
                    checkpoints[0], cp, overwrite=True
                ), f"Failed to copy {checkpoints[0]} to {cp}"

        write_timer.stop()
        logger.info(
            "Saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )

    if not end_of_epoch and cfg.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        if cfg.keep_interval_updates_pattern == -1:
            checkpoints = checkpoint_paths(
                cfg.save_dir, pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix)
            )
        else:
            checkpoints = checkpoint_paths(
                cfg.save_dir,
                pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix),
                keep_match=True,
            )
            checkpoints = [
                x[0]
                for x in checkpoints
                if x[1] % cfg.keep_interval_updates_pattern != 0
            ]

        for old_chk in checkpoints[cfg.keep_interval_updates :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    if cfg.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            cfg.save_dir, pattern=r"checkpoint(\d+){}\.pt".format(suffix)
        )
        for old_chk in checkpoints[cfg.keep_last_epochs :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    if cfg.keep_best_checkpoints > 0:
        # only keep the best N checkpoints according to validation metric
        checkpoints = checkpoint_paths(
            cfg.save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*){}\.pt".format(
                cfg.best_checkpoint_metric, suffix
            ),
        )
        if not cfg.maximize_best_checkpoint_metric:
            checkpoints = checkpoints[::-1]
        for old_chk in checkpoints[cfg.keep_best_checkpoints :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)



# from checkpoint_utils.py - cleanup to simplify.
def load_checkpoint(cfg: CheckpointConfig, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator.

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """

    reset_optimizer = cfg.reset_optimizer
    reset_lr_scheduler = cfg.reset_lr_scheduler
    optimizer_overrides = ast.literal_eval(cfg.optimizer_overrides)
    reset_meters = cfg.reset_meters
    reset_dataloader = cfg.reset_dataloader

    if cfg.finetune_from_model is not None and (
        reset_optimizer or reset_lr_scheduler or reset_meters or reset_dataloader
    ):
        raise ValueError(
            "--finetune-from-model can not be set together with either --reset-optimizer"
            " or reset_lr_scheduler or reset_meters or reset_dataloader"
        )

    suffix = trainer.checkpoint_suffix
    if (
        cfg.restore_file == "checkpoint_last.pt"
    ):  # default value of restore_file is 'checkpoint_last.pt'
        checkpoint_path = os.path.join(
            cfg.save_dir, "checkpoint_last{}.pt".format(suffix)
        )
        first_launch = not PathManager.exists(checkpoint_path)
        if cfg.finetune_from_model is not None and first_launch:
            # if there is no last checkpoint to restore, start the finetune from pretrained model
            # else just use usual logic to load checkpoint, e.g. restart from last checkpoint and etc.
            if PathManager.exists(cfg.finetune_from_model):
                checkpoint_path = cfg.finetune_from_model
                reset_optimizer = True
                reset_lr_scheduler = True
                reset_meters = True
                reset_dataloader = True
                logger.info(
                    f"loading pretrained model from {checkpoint_path}: "
                    "optimizer, lr scheduler, meters, dataloader will be reset"
                )
            else:
                raise ValueError(
                    f"--funetune-from-model {cfg.finetune_from_model} does not exist"
                )
    elif suffix is not None:
        checkpoint_path = cfg.restore_file.replace(".pt", suffix + ".pt")
    else:
        checkpoint_path = cfg.restore_file

    if cfg.restore_file != "checkpoint_last.pt" and cfg.finetune_from_model:
        raise ValueError(
            "--finetune-from-model and --restore-file (non-default value) "
            "can not be specified together: " + str(cfg)
        )

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        reset_optimizer,
        reset_lr_scheduler,
        optimizer_overrides,
        reset_meters=reset_meters,
    )

    if (
        extra_state is not None
        and "best" in extra_state
        and not reset_optimizer
        and not reset_meters
    ):
        save_checkpoint.best = extra_state["best"]

    if extra_state is not None and not reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state["train_iterator"]
        epoch_itr = trainer.get_train_iterator(
            epoch=itr_state["epoch"], load_dataset=True, **passthrough_args
        )
        epoch_itr.load_state_dict(itr_state)
    else:
        epoch_itr = trainer.get_train_iterator(
            epoch=1, load_dataset=True, **passthrough_args
        )

    trainer.lr_step(epoch_itr.epoch)

    return extra_state, epoch_itr



# note: scripts below for translation stage - can be simplified by avoiding the ensemble part or making it optional

#
# def load_model_ensemble(
#     filenames,
#     arg_overrides: Optional[Dict[str, Any]] = None,
#     task=None,
#     strict=True,
#     suffix="",
#     num_shards=1,
#     state=None,
# ):
#     """Loads an ensemble of models.
#
#     Args:
#         filenames (List[str]): checkpoint files to load
#         arg_overrides (Dict[str,Any], optional): override model args that
#             were used during model training
#         task (fairseq.tasks.FairseqTask, optional): task to use for loading
#     """
#     assert not (
#         strict and num_shards > 1
#     ), "Cannot load state dict with strict=True and checkpoint shards > 1"
#     ensemble, args, _task = load_model_ensemble_and_task(
#         filenames,
#         arg_overrides,
#         task,
#         strict,
#         suffix,
#         num_shards,
#         state,
#     )
#     return ensemble, args
#

# def get_maybe_sharded_checkpoint_filename(
#     filename: str, suffix: str, shard_idx: int, num_shards: int
# ) -> str:
#     orig_filename = filename
#     filename = filename.replace(".pt", suffix + ".pt")
#     fsdp_filename = filename[:-3] + f"-shard{shard_idx}.pt"
#     model_parallel_filename = orig_filename[:-3] + f"_part{shard_idx}.pt"
#     if PathManager.exists(fsdp_filename):
#         return fsdp_filename
#     elif num_shards > 1:
#         return model_parallel_filename
#     else:
#         return filename
#
#
# def load_model_ensemble_and_task(
#     filenames,
#     arg_overrides: Optional[Dict[str, Any]] = None,
#     task=None,
#     strict=True,
#     suffix="",
#     num_shards=1,
#     state=None,
# ):
#     assert state is None or len(filenames) == 1
#
#     from fairseq import tasks
#
#
#     assert not (
#         strict and num_shards > 1
#     ), "Cannot load state dict with strict=True and checkpoint shards > 1"
#     ensemble = []
#     cfg = None
#     for filename in filenames:
#         orig_filename = filename
#         model_shard_state = {"shard_weights": [], "shard_metadata": []}
#         assert num_shards > 0
#         st = time.time()
#         for shard_idx in range(num_shards):
#             filename = get_maybe_sharded_checkpoint_filename(
#                 orig_filename, suffix, shard_idx, num_shards
#             )
#
#             if not PathManager.exists(filename):
#                 raise IOError("Model file not found: {}".format(filename))
#             if state is None:
#                 state = load_checkpoint_to_cpu(filename, arg_overrides)
#             if "args" in state and state["args"] is not None:
#                 cfg = convert_namespace_to_omegaconf(state["args"])
#             elif "cfg" in state and state["cfg"] is not None:
#                 cfg = state["cfg"]
#             else:
#                 raise RuntimeError(
#                     f"Neither args nor cfg exist in state keys = {state.keys()}"
#                 )
#
#             if task is None:
#                 task = tasks.setup_task(cfg.task)
#
#             if "task_state" in state:
#                 task.load_state_dict(state["task_state"])
#
#             if "fsdp_metadata" in state and num_shards > 1:
#                 model_shard_state["shard_weights"].append(state["model"])
#                 model_shard_state["shard_metadata"].append(state["fsdp_metadata"])
#                 # check FSDP import before the code goes too far
#                 if not has_FSDP:
#                     raise ImportError(
#                         "Cannot find FullyShardedDataParallel. "
#                         "Please install fairscale with: pip install fairscale"
#                     )
#                 if shard_idx == num_shards - 1:
#                     consolidated_model_state = FSDP.consolidate_shard_weights(
#                         shard_weights=model_shard_state["shard_weights"],
#                         shard_metadata=model_shard_state["shard_metadata"],
#                     )
#                     model = task.build_model(cfg.model)
#                     model.load_state_dict(
#                         consolidated_model_state, strict=strict, model_cfg=cfg.model
#                     )
#             else:
#                 # model parallel checkpoint or unsharded checkpoint
#                 model = task.build_model(cfg.model)
#                 model.load_state_dict(
#                     state["model"], strict=strict, model_cfg=cfg.model
#                 )
#
#             # reset state so it gets loaded for the next model in ensemble
#             state = None
#             if shard_idx % 10 == 0 and shard_idx > 0:
#                 elapsed = time.time() - st
#                 logger.info(
#                     f"Loaded {shard_idx} shards in {elapsed:.2f}s, {elapsed / (shard_idx+1):.2f}s/shard"
#                 )
#
#         # build model for ensemble
#         ensemble.append(model)
#     return ensemble, cfg, task