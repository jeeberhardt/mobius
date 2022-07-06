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

    def __init__(self, acquisition_function, search_protocol):
        self._acq_fun = acquisition_function
        self._search_protocol = search_protocol

    def ask(self, batch_size=None):
        # Use the training set from the surrogate model as inputs for the optimization
        suggested_polymers = self._acq_fun.surrogate_model.X_train_original.copy()
        predicted_values = self._acq_fun.surrogate_model.y_train.copy()

        samplers = [s['function'](**s['parameters']) for name_sampler, s in self._search_protocol.items()]

        for sampler in samplers:
            suggested_polymers, predicted_values = sampler.run(self._acq_fun, suggested_polymers, predicted_values)

        # Sort sequences by scores in the decreasing order (best to worst)
        if self._acq_fun.goal == 'minimize':
            sorted_indices = np.argsort(predicted_values)
        else:
            sorted_indices = np.argsort(predicted_values)[::-1]

        suggested_polymers = suggested_polymers[sorted_indices]
        predicted_values = predicted_values[sorted_indices]

        return suggested_polymers[:batch_size], predicted_values[:batch_size]

    def tell(self, polymers, values):
        self._acq_fun.surrogate_model.fit(polymers, values)

    def recommand(self, polymers, values, batch_size=None):
        self.tell(polymers, values)
        suggested_polymers, predicted_values = self.ask(batch_size)

        return suggested_polymers, predicted_values

    def optimize(self, emulator, num_iter, batch_size):
        # Use the training set from the surrogate model as inputs for the optimization
        all_suggested_polymers = self._acq_fun.surrogate_model.X_train_original.copy()
        all_exp_values = self._acq_fun.surrogate_model.y_train.copy()

        for i in range(num_iter):
            suggested_polymers, predicted_values = self.recommand(all_suggested_polymers, all_exp_values, batch_size)

            suggested_polymers_fasta = [''.join(c.split('$')[0].split('{')[1].split('}')[0].split('.')) for c in suggested_polymers]
            exp_values = emulator.predict(suggested_polymers_fasta)
            
            all_suggested_polymers = np.concatenate([all_suggested_polymers, suggested_polymers])
            all_exp_values = np.concatenate([all_exp_values, exp_values])

        # Sort sequences by scores in the decreasing order (best to worst)
        if self._acq_fun.goal == 'minimize':
            sorted_indices = np.argsort(all_exp_values)
        else:
            sorted_indices = np.argsort(all_exp_values)[::-1]

        all_suggested_polymers = all_suggested_polymers[sorted_indices]
        all_exp_values = all_exp_values[sorted_indices]

        return all_suggested_polymers, all_exp_values
