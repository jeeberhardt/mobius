#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - termination criteria
#

from pymoo.core.termination import Termination


class NoChange(Termination):
    def __init__(self, only_feas=True, **kwargs):
        """
        Terminate if the best sequence does not change.

        """
        super().__init__(**kwargs)
        self.only_feas = only_feas
        self._current_best = set()

    def _update(self, algorithm):
        opt = algorithm.opt
        X = opt.get("X")

        if self.only_feas:
            X = X[opt.get("feas")]

        X = X.ravel()

        diff = set(X).symmetric_difference(self._current_best)

        print(diff)

        if len(diff) == 0:
            return 1.0
        else:
            self._current_best = set(X)
            return 0.0
