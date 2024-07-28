#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Display
#

from pymoo.util.display.column import Column
from pymoo.util.display.output import Output, pareto_front_if_possible


class MinimumConstraintViolation(Column):

    def __init__(self, **kwargs) -> None:
        super().__init__("cv_min", **kwargs)

    def update(self, algorithm):
        self.value = algorithm.opt.get("cv").min()


class AverageConstraintViolation(Column):

    def __init__(self, **kwargs) -> None:
        super().__init__("cv_avg", **kwargs)

    def update(self, algorithm):
        self.value = algorithm.pop.get("cv").mean()


class SingleObjectiveOutput(Output):

    def __init__(self):
        super().__init__()
        self.cv_min = MinimumConstraintViolation()
        self.cv_avg = AverageConstraintViolation()

        self.f_min = Column(name="f_min", width=20)
        self.f_avg = Column(name="f_avg", width=20)
        self.f_gap = Column(name="f_gap", width=20)
        self.best = Column(name="best sequence", width=40)

    def initialize(self, algorithm):
        problem = algorithm.problem

        if problem.has_constraints():
            self.columns += [self.cv_min, self.cv_avg]

        self.columns += [self.f_avg, self.f_min]
        self.columns += [self.best]

    def update(self, algorithm):
        super().update(algorithm)

        f, cv, feas = algorithm.pop.get("f", "cv", "feas")

        if feas.sum() > 0:
            self.f_avg.set(f[feas].mean())
        else:
            self.f_avg.set(None)

        opt = algorithm.opt[0]

        if opt.feas:
            self.f_min.set(opt.f)
        else:
            self.f_min.set(None)
        
        if len(opt.x[0]) > 40:
            self.best.set(f'{opt.x[0][:18]}....{opt.x[0][-18:]}')
        else:
            self.best.set(opt.x[0])
