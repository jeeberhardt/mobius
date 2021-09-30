#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - virtual target
#

import os

import numpy as np

from . import utils


def _reversed_lj_coefficient(epsilon, reqm, a, b):
    """Compute coefficients."""
    return (np.abs(b - a)) / a * epsilon * reqm**b


def _smooth_distance(r, reqm, smooth=0.5):
        """Smooth distance."""
        # Otherwise we changed r in-place.
        r = r.copy()
        sf = .5 * smooth
        r[((reqm - sf) < r) & (r < (reqm + sf))] = reqm
        r[r >= reqm + sf] -= sf
        r[r <= reqm - sf] += sf
        return r


class ForceField:
    """Class to handle the forcefield"""
    def __init__(self, parameter_file=None):
        """Initialize the peptide forcefield

        Args:
            parameter_file (str): parameter file (default: None; take the parameter file from data directory)

        """
        if parameter_file is None:
            parameter_file = os.path.join(utils.path_module('mobius'), 'data/parameters.csv')

        dtype = [('AA3', 'U3'), ('AA1', 'U1'), ('hydrophilicity', 'f4'), ('volume', 'f4')]
        self._parameters = np.genfromtxt(parameter_file, dtype=dtype, delimiter=',', skip_header=1)

    def parameters(self):
        """Return all the parameters for each amino acid"""
        return self._parameters

    def _volume(self, v_residue, v_pharmacophore, epsilon=1, smooth=0.1, n=9, m=3):
        """Score the residue based on its volume compared to the pharmacophore

        A "reversed" Lennard-Jones potential is used for the volume term:

            V_rLJ = (r**n / A) - (r**m / B)

        Args:
            v_residue (float): volume of the residue
            v_pharmacophore (float): volume of the pharmacophore at that position
            epsilon (float): depth of the reversed LJ potential
            smooth (float): smoothing factor like in AutoDock
            n (int): repulsive term
            m (int): attractive term

        Returns:
            float: score of the volume term

        """
        A = _reversed_lj_coefficient(epsilon, v_pharmacophore, m, n)
        B = _reversed_lj_coefficient(epsilon, v_pharmacophore, n, m)

        if smooth > 0:
            v_residue = _smooth_distance(v_residue, v_pharmacophore, smooth)

        # We add epsilon so V is equal to zero when v_residue == v_pharmacophore
        score = epsilon + (((v_residue**n) / A)) - (((v_residue**m) / B))

        return score

    def _hydrophilicity(self, h_residue, h_pharmacophore, k=10):
        """Calculate the hydrophilic score

        Use a quadratic function to score

        Args:
            h_residue (float): hydrophilicity of the residue
            h_pharmacophore (float): hydrophilicity at that position in the pharmacophore
            k (int): factor

        Returns:
            float: score for the hydrophilic term

        """
        return k * (h_residue - h_pharmacophore)**2

    def _desolvation(self, h_residue, se_pharmacophore, k=10):
        """Calculate the desolvation cost

        A desolvation cost is paid only if the solvent exposure is higher than
        the hydrophilicity of the residue. For example:
            - Solvent_exposure = 1 and hydrophilicity = 0 --> cost: 1 witk k = 1
            - Solvent exposure = 1 and hydrophilicity = 1 --> cost: 0 with k = 1
            - Solvent exposure = 0 and hydrophilicity = 1 --> cost: 0 because exposure < hydrophilicity

        Args:
            h_residue (float): hydrophilicity of the residue
            se_pharmacophore (float): solvent exposure at that position in the pharmacophore
            k (int): factor

        Returns:
            float: score for the desolvation term

        """
        if h_residue < se_pharmacophore:
            return k * (se_pharmacophore - h_residue)**2
        else:
            return 0.

    def score(self, pharmacophore, peptide, details=False):
        """Score the peptide sequence based on the provded pharmacophore

        Args:
            pharmacophore (structured array): pharmacophore containing the solvent_exposure, hydrophilicity and
                volume information
            peptide (str): peptide sequence to score
            details (bool): if True, returns also the score per residue (default: False)

        Returns:
            float: score of the peptide
            (float, ndarray): score of the peptide and score per residue, if details = True

        """
        assert len(pharmacophore) == len(peptide), "The pharmacophore and peptide have different size."

        score_residues = []

        for i in range(len(pharmacophore)):
            param_residue = self._parameters[self._parameters['AA1'] == peptide[i]]
            param_pharmacophore = pharmacophore[i]

            """
            print('Pharmacophore - V: %12.3f / H: %12.3f / SE: %12.3f' % (param_pharmacophore['volume'],
                                                                          param_pharmacophore['hydrophilicity'],
                                                                          param_pharmacophore['solvent_exposure']))
            print('Residue %s     - V: %12.3f / H: %12.3f' % (peptide[i], param_residue['volume'],
                                                              param_residue['hydrophilicity']))
            """

            # Score of the volume, hydrophilicity and desolvation terms
            score_volume = self._volume(param_residue['volume'], param_pharmacophore['volume'])
            score_hydrophilicity = self._hydrophilicity(param_residue['hydrophilicity'], param_pharmacophore['hydrophilicity'])
            score_desolvation = self._desolvation(param_residue['hydrophilicity'], param_pharmacophore['solvent_exposure'])

            """The volume and hydrophicility terms are weighted by the buriedness. More buried is the residue, 
            stronger are the interactions between the residue and the pharmacophore. In contrary, a completely 
            exposed residue (solvent_exposure = 1) is not interacting with the pharmacophore, so 
            (volume + hydrophilicity) = 0. However there is a desolvation cost to pay if the residue is completely 
            hydrophobic.

            Examples:
                Buried pocket (solvent exposure = 0)           --> score = volume + hydrophilicity
                Solvent exposed (solvent exposure = 1)         --> score = desolvation
                Pocket on the surface (solvant exposure = 0.5) --> score = 0.5 * (volume + hydrophilicity) + desolvation

            """
            buriedness = 1 - param_pharmacophore['solvent_exposure']
            score_residue = buriedness * (score_volume + score_hydrophilicity) + score_desolvation

            #print('Score         - V: %12.3f / H: %12.3f / D : %12.3f' % (score_volume, score_hydrophilicity, score_desolvation))

            score_residues.extend(score_residue)

        #print("")

        score_residues = np.array(score_residues)
        score = np.sum(score_residues)

        if details:
            return score, score_residues
        else:
            return score
