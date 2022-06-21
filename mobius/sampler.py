#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - polymer sampler
#

from abc import ABC, abstractmethod


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


def PolymerSampler(_Sampler):

    def __init__(self, surrogate_model, search_protocol, acquisition_function, goal='minimize'):
        self._surrogate_model = surrogate_model
        self._search_protocol = search_protocol
        self._acquisition_function = acquisition_function
        self._goal = goal

    def ask(self, batch_size=None):
        pass

    def tell(self, data, value):
        pass

    def recommand(self, data, value, batch_size=None):
        pass

    def optimize(self, emulator, num_iter, batch_size):
        pass
