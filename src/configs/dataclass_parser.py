"""
Script implements two types of converting namespace to args
- flat: similar with argparse (available in v1.0
- structured: using omega config (available in future version) - commented out now - check necessity first?
"""

# from dataclass/utils.py
from argparse import ArgumentError, ArgumentParser, Namespace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type
from omegaconf import II, MISSING
import inspect
import re, ast

from .dataclass import DataclassBase
from src.utils.helpers import eval_str_list, interpret_dc_type

# from dataclass/utils - find proper place
from omegaconf import DictConfig, OmegaConf, open_dict, _utils
from argparse import ArgumentError, ArgumentParser, Namespace
from src.configs.dataclass import DataclassConfig


# should go to utils/data.py - used by any script that changes dataclass to parser args
# note: should probabaly go to separate script under configs (dataclass_argparse.py?)
def gen_parser_from_dataclass(
    parser: ArgumentParser,
    dataclass_instance: DataclassBase,
    delete_default: bool = False,
    with_prefix: Optional[str] = None,
) -> None:
    """
        convert a dataclass instance to tailing parser arguments.

        If `with_prefix` is provided, prefix all the keys in the resulting parser with it. It means that we are
        building a flat namespace from a structured dataclass (see transformer_config.py for example).
    """

    def argparse_name(name: str):
        if name == "data" and (with_prefix is None or with_prefix == ''):
            # normally data is positional args, so we don't add the -- nor the prefix
            return name
        if name == "_name":
            # private member, skip
            return None
        full_name = "--" + name.replace("_", "-")
        if with_prefix is not None and with_prefix != '':
            # if a prefix is specified, construct the prefixed arg name
            full_name = with_prefix + "-" + full_name[2:]  # strip -- when composing
        return full_name

    def get_kwargs_from_dc(dataclass_instance: DataclassBase, k: str) -> Dict[str, Any]:
        """k: dataclass attributes"""
        kwargs = {}

        field_type = dataclass_instance._get_type(k)
        inter_type = interpret_dc_type(field_type)

        field_default = dataclass_instance._get_default(k)

        if isinstance(inter_type, type) and issubclass(inter_type, Enum):
            field_choices = [t.value for t in list(inter_type)]
        else:
            field_choices = None

        field_help = dataclass_instance._get_help(k)
        field_const = dataclass_instance._get_argparse_const(k)

        if isinstance(field_default, str) and field_default.startswith("${"):
            kwargs["default"] = field_default
        else:
            if field_default is MISSING:
                kwargs["required"] = True
            if field_choices is not None:
                kwargs["choices"] = field_choices
            if (
                isinstance(inter_type, type)
                and (issubclass(inter_type, List) or issubclass(inter_type, Tuple))
            ) or ("List" in str(inter_type) or "Tuple" in str(inter_type)):
                if "int" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, int)
                elif "float" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, float)
                elif "str" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, str)
                else:
                    raise NotImplementedError(
                        "parsing of type " + str(inter_type) + " is not implemented"
                    )
                if field_default is not MISSING:
                    kwargs["default"] = (
                        ",".join(map(str, field_default))
                        if field_default is not None
                        else None
                    )
            elif (
                isinstance(inter_type, type) and issubclass(inter_type, Enum)
            ) or "Enum" in str(inter_type):
                kwargs["type"] = str
                if field_default is not MISSING:
                    if isinstance(field_default, Enum):
                        kwargs["default"] = field_default.value
                    else:
                        kwargs["default"] = field_default
            elif inter_type is bool:
                kwargs["action"] = (
                    "store_false" if field_default is True else "store_true"
                )
                kwargs["default"] = field_default
            else:
                kwargs["type"] = inter_type
                if field_default is not MISSING:
                    kwargs["default"] = field_default

        # build the help with the hierarchical prefix
        if with_prefix is not None and with_prefix != '' and field_help is not None:
            field_help = with_prefix[2:] + ': ' + field_help

        kwargs["help"] = field_help
        if field_const is not None:
            kwargs["const"] = field_const
            kwargs["nargs"] = "?"

        return kwargs

    for k in dataclass_instance._get_all_attributes():
        field_name = argparse_name(dataclass_instance._get_name(k))
        field_type = dataclass_instance._get_type(k)
        if field_name is None:
            continue
        elif inspect.isclass(field_type) and issubclass(field_type, DataclassBase):
            # for fields that are of type FairseqDataclass, we can recursively
            # add their fields to the namespace (so we add the args from model, task, etc. to the root namespace)
            prefix = None
            if with_prefix is not None:
                # if a prefix is specified, then we don't want to copy the subfields directly to the root namespace
                # but we prefix them with the name of the current field.
                prefix = field_name
            gen_parser_from_dataclass(parser, field_type(), delete_default, prefix)
            continue

        kwargs = get_kwargs_from_dc(dataclass_instance, k)

        field_args = [field_name]
        alias = dataclass_instance._get_argparse_alias(k)
        if alias is not None:
            field_args.append(alias)

        if "default" in kwargs:
            if isinstance(kwargs["default"], str) and kwargs["default"].startswith(
                "${"
            ):
                if kwargs["help"] is None:
                    # this is a field with a name that will be added elsewhere
                    continue
                else:
                    del kwargs["default"]
            if delete_default and "default" in kwargs:
                del kwargs["default"]
        try:
            parser.add_argument(*field_args, **kwargs)
        except ArgumentError:
            pass



#
# def override_module_args(args: Namespace) -> Tuple[List[str], List[str]]:
#     """use the field in args to overrides those in cfg"""
#     overrides = []
#     deletes = []
#
#     for k in FairseqConfig.__dataclass_fields__.keys():
#         overrides.extend(
#             _override_attr(k, FairseqConfig.__dataclass_fields__[k].type, args)
#         )
#
#     if args is not None:
#         if hasattr(args, "task"):
#             from fairseq.tasks import TASK_DATACLASS_REGISTRY
#
#             migrate_registry(
#                 "task", args.task, TASK_DATACLASS_REGISTRY, args, overrides, deletes
#             )
#         else:
#             deletes.append("task")
#
#         # these options will be set to "None" if they have not yet been migrated
#         # so we can populate them with the entire flat args
#         CORE_REGISTRIES = {"criterion", "optimizer", "lr_scheduler"}
#
#         from fairseq.registry import REGISTRIES
#
#         for k, v in REGISTRIES.items():
#             if hasattr(args, k):
#                 migrate_registry(
#                     k,
#                     getattr(args, k),
#                     v["dataclass_registry"],
#                     args,
#                     overrides,
#                     deletes,
#                     use_name_as_val=k not in CORE_REGISTRIES,
#                 )
#             else:
#                 deletes.append(k)
#
#         no_dc = True
#         if hasattr(args, "arch"):
#             from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_MODEL_NAME_REGISTRY
#
#             if args.arch in ARCH_MODEL_REGISTRY:
#                 m_cls = ARCH_MODEL_REGISTRY[args.arch]
#                 dc = getattr(m_cls, "__dataclass", None)
#                 if dc is not None:
#                     m_name = ARCH_MODEL_NAME_REGISTRY[args.arch]
#                     overrides.append("model={}".format(m_name))
#                     overrides.append("model._name={}".format(args.arch))
#                     # override model params with those exist in args
#                     overrides.extend(_override_attr("model", dc, args))
#                     no_dc = False
#         if no_dc:
#             deletes.append("model")
#
#     return overrides, deletes
#
#
# def convert_namespace_to_omegaconf(args: Namespace) -> DictConfig:
#     """Convert a flat argparse.Namespace to a structured DictConfig."""
#
#     # Here we are using field values provided in args to override counterparts inside config object
#     overrides, deletes = override_module_args(args)
#
#     # configs will be in fairseq/config after installation
#     config_path = os.path.join("..", "config")
#
#     #GlobalHydra.instance().clear() # a similar heirarchical config mngmt - not sure why its here
#
#     with initialize(config_path=config_path):
#         try:
#             composed_cfg = compose("config", overrides=overrides, strict=False)
#         except:
#             logger.error("Error when composing. Overrides: " + str(overrides))
#             raise
#
#         for k in deletes:
#             composed_cfg[k] = None
#
#     cfg = OmegaConf.create(
#         OmegaConf.to_container(composed_cfg, resolve=True, enum_to_str=True)
#     )
#
#     # hack to be able to set Namespace in dict config. this should be removed when we update to newer
#     # omegaconf version that supports object flags, or when we migrate all existing models
#     #from omegaconf import _utils
#
#     # with omegaconf_no_object_check():
#     #     if cfg.task is None and getattr(args, "task", None):
#     #         cfg.task = Namespace(**vars(args))
#     #         from fairseq.tasks import TASK_REGISTRY
#     #
#     #         _set_legacy_defaults(cfg.task, TASK_REGISTRY[args.task])
#     #         cfg.task._name = args.task
#     #     if cfg.model is None and getattr(args, "arch", None):
#     #         cfg.model = Namespace(**vars(args))
#     #         from fairseq.models import ARCH_MODEL_REGISTRY
#     #
#     #         _set_legacy_defaults(cfg.model, ARCH_MODEL_REGISTRY[args.arch])
#     #         cfg.model._name = args.arch
#     #     if cfg.optimizer is None and getattr(args, "optimizer", None):
#     #         cfg.optimizer = Namespace(**vars(args))
#     #         from fairseq.optim import OPTIMIZER_REGISTRY
#     #
#     #         _set_legacy_defaults(cfg.optimizer, OPTIMIZER_REGISTRY[args.optimizer])
#     #         cfg.optimizer._name = args.optimizer
#     #     if cfg.lr_scheduler is None and getattr(args, "lr_scheduler", None):
#     #         cfg.lr_scheduler = Namespace(**vars(args))
#     #         from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY
#     #
#     #         _set_legacy_defaults(
#     #             cfg.lr_scheduler, LR_SCHEDULER_REGISTRY[args.lr_scheduler]
#     #         )
#     #         cfg.lr_scheduler._name = args.lr_scheduler
#     #     if cfg.criterion is None and getattr(args, "criterion", None):
#     #         cfg.criterion = Namespace(**vars(args))
#     #         from fairseq.criterions import CRITERION_REGISTRY
#     #
#     #         _set_legacy_defaults(cfg.criterion, CRITERION_REGISTRY[args.criterion])
#     #         cfg.criterion._name = args.criterion
#
#     OmegaConf.set_struct(cfg, True)
#     return cfg

