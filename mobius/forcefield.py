#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - virtual target
#

import os

import numpy as np
import pandas as pd

from . import utils


class ForceField:
    """Class to handle the forcefield"""
    def __init__(self, parameter_file=None):
        if parameter_file is None:
            self._parameters = os.path.join(utils.path_module('mobius'), 'data/parameters.csv')

        self._parameters = pd.read_csv(parameter_file)

    def parameters(self):
        return self._parameters

    def score(self, pharmacophore, peptide, details=False):
        assert len(pharmacophore) == len(peptide), "The pharmacophore and peptide have different size."

        score_residues = []

        for i in range(len(pharmacophore)):
            score_residue = 0
            param_residue = self._parameters[self._parameters['AA1'] == peptide[i]]
            param_pharmac = pharmacophore.loc[i]

            #print(param_residue['volume'].values)

            if param_pharmac['solvent_exposure'] < 0.8:
                # Score volume
                if param_residue['volume'].values[0] < param_pharmac['volume']:
                    score_residue += (param_residue['volume'] - param_pharmac['volume'])**2
                else:
                    score_residue += (param_residue['volume'] - param_pharmac['volume'])**2 + np.exp(param_residue['volume'] - param_pharmac['volume']) - 1

            # Score hydrophilicity
            score_residue += (param_residue['hydrophilicity'] - param_pharmac['hydrophilicity'])**2

            score_residues.extend(score_residue)

        score = np.sum(score_residues)

        print(score, score_residues)

        if details:
            return score, score_residues
        else:
            return score
