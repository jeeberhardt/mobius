#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Dummy Model
#

from abc import ABC, abstractmethod

import numpy as np


class _SurrogateModel(ABC):

    @property
    @abstractmethod
    def X_train(self):
        pass

    @property
    @abstractmethod
    def y_train(self):
        pass

    @abstractmethod
    def fit(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()

    @abstractmethod
    def score(self):
        raise NotImplementedError()


class DummyModel(_SurrogateModel):

    def __init__(self):
        self._kernel = None
        self._input_transformer = None
        self._likelihood = None
        self._model = None
        self._X_train = np.array([])
        self._y_train = np.array([])

    @property
    def X_train(self):
        return self._X_train

    @property
    def y_train(self):
        return self._y_train

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        return None, None

    def score(self, X_test, y_test):
        return None
