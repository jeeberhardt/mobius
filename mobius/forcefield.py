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

        dtype = [('AA3', 'U3'), ('AA1', 'U1'), ('hydrophilicity', 'f4'),
                 ('volume', 'f4'), ('net_charge', 'i4'),
                 ('hb_don', 'i4'), ('hb_acc', 'i4'), ('hb_don_acc', 'i4')]
        self._parameters = np.genfromtxt(parameter_file, dtype=dtype, delimiter=',', skip_header=1)
        self._term_parameters = {'vdw': {"epsilon": 1,
                                         'smooth': 0.1,
                                         'score_max': None,
                                         'n': 9,
                                         'm': 3}}

    def parameters(self):
        """Return all the parameters for each amino acid"""
        return self._parameters

    def _van_der_waals(self, v_residue, v_pharmacophore, epsilon=1, smooth=0.1, n=9, m=3, score_max=None):
        """Calculate the score of the van der Waals-like term

        A "reversed" Lennard-Jones potential is used for the volume term:

            V_rLJ = (r**n / A) - (r**m / B)

        Args:
            v_residue (float): volume of the residue
            v_pharmacophore (float): volume of the pharmacophore
            epsilon (float): depth of the reversed LJ potential
            smooth (float): smoothing factor like in AutoDock
            n (int): repulsive term
            m (int): attractive term
            score_max (float): truncate energy (default: None)

        Returns:
            float: score of the vdW term (between 0 and inf)

        """
        A = _reversed_lj_coefficient(epsilon, v_pharmacophore, m, n)
        B = _reversed_lj_coefficient(epsilon, v_pharmacophore, n, m)

        if smooth > 0:
            v_residue = _smooth_distance(v_residue, v_pharmacophore, smooth)

        # We add epsilon so V is equal to zero when v_residue == v_pharmacophore
        score = epsilon + (((v_residue**n) / A)) - (((v_residue**m) / B))

        try:
            score = score_max if score > score_max else score
        except:
            pass

        return score

    def _electrostatic(self, h_residue, h_pharmacophore, c_residue, c_phamacophore):
        """Calculate the score of the electrostatic-like term

        Cumulative sum of:
            - a quadratic function for the hydrophilicity (score between 0 and 1)
            - an absolute sum of the net charges (score between 0 and 1)

        Net_charge scoring:
            c_1 =  0  -  c_2 =  0  --> 0
            c_1 =  1  -  c_2 =  1  --> 0
            c_1 = -1  -  c_2 = -1  --> 0
            c_1 =  0  -  c_2 =  1  --> 1
            c_1 =  1  -  c_2 =  0  --> 1
            c_1 =  1  -  c_2 = -1  --> 2

        Args:
            h_residue (float): hydrophilicity of the residue
            h_pharmacophore (float): hydrophilicity of the pharmacophore
            c_residue (int): net charge of the residue
            c_phamacophore (int): net charge of the pharmacophore

        Returns:
            float: score of the electrostatic term (between 0 and 2)

        """
        score = (h_residue - h_pharmacophore)**2
        score += np.abs(c_residue - c_phamacophore)

        return score

    def _hydrogen_bond(self, hb_residue, hb_pharmacophore):
        """Calculate the score for the hydrogen bond-like term

        Scoring:
            - A     ## D                 --> 0
            - A + D ## A + D             --> 0
            - AD    ## AD                --> 0
            - A     ## A or AD or A + D  --> 1 (because 1 unsatisfied HB)
            - AD    ## A or D or A + D   --> 1
            - A     ## None              --> 1
            - AD    ## None              --> 2 (because 2 unsatisfied HB)
            - A + D ## None              --> 2

        Note: A + D (Asn, Gln) != AD (Thr, Ser, Cys, Tyr)

        Args:
            hb_residue (list): hb parameters of residue [hb_don, hb_acc, hb_don_acc]
            hb_pharmacophore (list): hb parameters of pharmacophore [hb_don, hb_acc, hb_don_acc]

        Returns:
            int: score of the hydrogen bond term (between 0 and 2)

        """
        if all([hb_residue[2], hb_pharmacophore[2]]):
            # If both are hb_donor_acceptor
            return 0
        elif any([hb_residue[2], hb_pharmacophore[2]]):
            # If one of the two is hb_donor_acceptor
            if np.sum([hb_residue[:2], hb_pharmacophore[:2]]) > 0:
                # It means that one is donor_acceptor and the other
                # one is donor or acceptor or (donor, acceptor)
                return 1
            else:
                # It means that one is donor_acceptor and the other
                # one is neither donor or acceptor
                return 2
        else:
            # HB donor
            score = np.abs(hb_residue[0] - hb_pharmacophore[0])
            # HB acceptor
            score += np.abs(hb_residue[1] - hb_pharmacophore[1])
            return score

    def _desolvation(self, h_residue, se_pharmacophore):
        """Calculate the score of the desolvation cost

        The desolvation cost is the hydrophobicity (1 - hydrophilicity) of that 
        residue weighted by the solvent exposure. For example:
            - Solvent_exposure = 1 and hydrophobicity = 1      --> cost: 1
            - Solvent exposure = 1 and hydrophobicity = 0      --> cost: 0
            - Solvent exposure = 0 and hydrophobicity = 0 or 1 --> cost: 0

        Args:
            h_residue (float): hydrophilicity of the residue
            se_pharmacophore (float): solvent exposure of the pharmacophore

        Returns:
            float: score of the desolvation term (between 0 and 1)

        """
        return se_pharmacophore * (1 - h_residue)

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
        score_terms = np.zeros(4)

        #print('Peptide %s' % peptide)

        for i in range(len(pharmacophore)):
            param_residue = self._parameters[self._parameters['AA1'] == peptide[i]]
            param_pharmacophore = pharmacophore[i]

            # vdW
            score_vdw = self._van_der_waals(param_residue['volume'], param_pharmacophore['volume'],
                                            epsilon=self._term_parameters['vdw']['epsilon'],
                                            smooth=self._term_parameters['vdw']['smooth'],
                                            n=self._term_parameters['vdw']['n'],
                                            m=self._term_parameters['vdw']['m'],
                                            score_max=self._term_parameters['vdw']['score_max'])

            # Electrostatic
            score_electrostatic = self._electrostatic(param_residue['hydrophilicity'], param_pharmacophore['hydrophilicity'],
                                                      param_residue['net_charge'], param_pharmacophore['net_charge'])

            # HBond
            hb_residue = param_residue[['hb_don', 'hb_acc', 'hb_don_acc']].item()
            hb_pharmacophore = param_pharmacophore[['hb_don', 'hb_acc', 'hb_don_acc']].item()
            score_hydrogen_bond = self._hydrogen_bond(hb_residue, hb_pharmacophore)

            # Desolvation
            score_desolvation = self._desolvation(param_residue['hydrophilicity'], param_pharmacophore['solvent_exposure'])

            """The vdW and electrostatic terms are weighted by the buriedness. More buried is the residue, 
            stronger are the interactions between the residue and the pharmacophore. In contrary, a completely 
            exposed residue (solvent_exposure = 1) is not interacting with the pharmacophore, so 
            (vdW + electrostatic) = 0. However there is a desolvation cost to pay if the residue is completely 
            hydrophobic.

            Examples:
                Buried pocket (solvent exposure = 0)           --> score = vdW + electrostatic + hydrogen_bond
                Solvent exposed (solvent exposure = 1)         --> score = desolvation
                Pocket on the surface (solvant exposure = 0.5) --> score = 0.5 * (vdW + electrostatic + hydrogen_bond) + desolvation

            """
            buriedness = 1 - param_pharmacophore['solvent_exposure']
            score_residue = buriedness * (score_vdw + score_electrostatic + score_hydrogen_bond) + score_desolvation

            #print('Residue %s     - %s' % (peptide[i], param_residue))
            #print('Score         - V: %12.3f / E: %12.3f / HB: %12.3f / D : %12.3f' % (score_vdw, score_electrostatic, score_hydrogen_bond, score_desolvation))

            score_residues.extend(score_residue)
            score_terms[0] += buriedness * score_vdw
            score_terms[1] += buriedness * score_electrostatic
            score_terms[2] += buriedness * score_hydrogen_bond
            score_terms[3] += score_desolvation

        #print("")

        score_residues = np.array(score_residues)
        score = np.sum(score_residues)

        if details:
            return score, score_terms #score_residues
        else:
            return score
