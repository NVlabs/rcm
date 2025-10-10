# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from omegaconf import OmegaConf

from imaginaire.lazy_config.instantiate import instantiate
from imaginaire.lazy_config.lazy import LazyCall, LazyConfig, LazyDict
from imaginaire.lazy_config.omegaconf_patch import to_object

OmegaConf.to_object = to_object

PLACEHOLDER = None

__all__ = ["PLACEHOLDER", "LazyCall", "LazyConfig", "LazyDict", "instantiate"]


DOC_BUILDING = os.getenv("_DOC_BUILDING", False)  # set in docs/conf.py


def fixup_module_metadata(module_name, namespace, keys=None):
    """
    Fix the __qualname__ of module members to be their exported api name, so
    when they are referenced in docs, sphinx can find them. Reference:
    https://github.com/python-trio/trio/blob/6754c74eacfad9cc5c92d5c24727a2f3b620624e/trio/_util.py#L216-L241
    """
    if not DOC_BUILDING:
        return
    seen_ids = set()

    def fix_one(qualname, name, obj):
        # avoid infinite recursion (relevant when using
        # typing.Generic, for example)
        if id(obj) in seen_ids:
            return
        seen_ids.add(id(obj))

        mod = getattr(obj, "__module__", None)
        if mod is not None and (mod.startswith(module_name) or mod.startswith("fvcore.")):
            obj.__module__ = module_name
            # Modules, unlike everything else in Python, put fully-qualitied
            # names into their __name__ attribute. We check for "." to avoid
            # rewriting these.
            if hasattr(obj, "__name__") and "." not in obj.__name__:
                obj.__name__ = name
                obj.__qualname__ = qualname
            if isinstance(obj, type):
                for attr_name, attr_value in obj.__dict__.items():
                    fix_one(objname + "." + attr_name, attr_name, attr_value)

    if keys is None:
        keys = namespace.keys()
    for objname in keys:
        if not objname.startswith("_"):
            obj = namespace[objname]
            fix_one(objname, objname, obj)


fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
