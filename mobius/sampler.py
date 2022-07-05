#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - polymer sampler
#

from abc import ABC, abstractmethod

import numpy as np

from .helm_genetic_operators import HELMGeneticOperators


class _Sampler(ABC):

    @abstractmethod
    def ask(self):
        raise NotImplementedError()

    @abstractmethod
    def tell(self):
        raise NotImplementedError()

    @abstractmethod
    def recommand(self):
        raise NotImplementedError()

    @abstractmethod
    def optimize(self):
        raise NotImplementedError()


class PolymerSampler(_Sampler):

    def __init__(self, surrogate_model, acquisition_function, search_protocol, goal='minimize'):
        assert goal in ['minimize', 'maximize'], 'The goal can only be \'minimize\' or \'maximize\'.'

        self._surrogate_model = surrogate_model
        self._search_protocol = search_protocol
        self._acquisition_function = acquisition_function
        self._goal = goal

    def ask(self, polymers, values, batch_size=None):
        samplers = [s['function'](**s['parameters']) for name_sampler, s in self._search_protocol.items()]
        acq_fun = self._acquisition_function(self._surrogate_model, values, self._goal)

        # Copy the input polymers
        suggested_polymers = np.array(polymers).copy()
        predicted_values = np.array(values).copy()

        for sampler in samplers:
            suggested_polymers, predicted_values = sampler.run(acq_fun, suggested_polymers, predicted_values)

        # Sort sequences by scores in the decreasing order (best to worst)
        if self._goal == 'minimize':
            sorted_indices = np.argsort(predicted_values)
        else:
            sorted_indices = np.argsort(predicted_values)[::-1]

        suggested_polymers = suggested_polymers[sorted_indices]
        predicted_values = predicted_values[sorted_indices]

        return suggested_polymers[:batch_size], predicted_values[:batch_size]

    def tell(self, polymers, values):
        self._surrogate_model.fit(polymers, values)

    def recommand(self, polymers, values, batch_size=None):
        self.tell(polymers, values)
        suggested_polymers = self.ask(polymers, values, batch_size)

    def optimize(self, emulator, num_iter, batch_size):
        pass
