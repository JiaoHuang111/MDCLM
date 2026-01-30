# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import glob
import importlib
import os
import sys

from omegaconf import DictConfig

from byprot.utils import import_modules

DATAMODULE_REGISTRY = {}


def register_datamodule(name):
    def decorator(cls):
        DATAMODULE_REGISTRY[name] = cls
        return cls

    return decorator


import_modules(os.path.dirname(__file__), "byprot.datamodules")

# ✅ 自动将所有导入的子模块注册为包属性（关键修复）
_pkg = sys.modules[__name__]
for modname in list(sys.modules.keys()):
    if modname.startswith("byprot.datamodules.") and modname != "byprot.datamodules":
        shortname = modname.split(".")[-1]
        setattr(_pkg, shortname, sys.modules[modname])

