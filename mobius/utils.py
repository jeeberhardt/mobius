
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - utils
#

from importlib import util


def path_module(module_name):
    specs = util.find_spec(module_name)
    if specs is not None:
        return specs.submodule_search_locations[0]
    return None
