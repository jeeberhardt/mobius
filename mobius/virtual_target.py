#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - virtual target
#

import numpy as np
import pandas as pd


def _find_optimal_sequence(forcefield, pharmacophore):
    scores = []
    residue_names = forcefield.parameters()['AA1']

    # Score each residue at each position
    for residue_name in residue_names:
        sequence = [residue_name] * pharmacophore.shape[0]
        _, score_details = forcefield.score(pharmacophore, sequence, True) 
        scores.append(score_details)

    # Get the index of the residues with the best (lowest) score
    residue_indices = np.argmin(scores, axis=0)
    # Get the residues names based on the index
    optimal_sequence = ''.join(residue_names[residue_indices])

    return optimal_sequence


class VirtualTarget:
    """Class to handle a virtual target"""
    def __init__(self, forcefield, seed=None):
        """Initialize the virtual target

        Args:
            forcefield (Forcefield): a forcefield to score interaction between
                a sequence and the virtual target
            seed (int): random seed

        """
        self._forcefield = forcefield
        self._sequence_length = None
        self._pharmacophore = None
        self._random_seed = seed
        self._rng = np.random.default_rng(self._random_seed)

    def __repr__(self):
        repr_str = 'Pharmacophore:\n'
        repr_str += '%s\n\n' % (pd.DataFrame(self._pharmacophore))
        repr_str += 'Optimal sequence: %s (score: %8.3f) \n\n' % (self._optimal_sequence, self.score_peptides([self._optimal_sequence]))
        repr_str += 'Parameters for the optimal sequence:\n'

        parameters = self._forcefield.parameters()
        data = []
        for s in self._optimal_sequence:
            data.append(parameters[parameters['AA1'] == s])
        repr_str += '%s' % (pd.DataFrame(np.hstack(data)))

        return repr_str

    def load_pharmacophore(self, input_filename):
        """Load the virtual phamacophore

        Args:
            input_filename (str): input csv filename containing the virtual
                pharmacophore

        """
        dtype = [('solvent_exposure', 'f4'), ('hydrophilicity', 'f4'), ('volume', 'f4')]
        self._pharmacophore = np.genfromtxt(input_filename, dtype=dtype, delimiter=',', skip_header=1)
        # Get the optimal sequence (sequence with the lowest score)
        self._optimal_sequence = _find_optimal_sequence(self._forcefield, self._pharmacophore)

    def optimal_sequence(self):
        """Return the optimal sequence for the current pharmacophore

        Returns:
            str: optimal sequence

        """
        return self._optimal_sequence

    def generate_random_pharmacophore(self, sequence_length):
        """Generate a random pharmacophore

        Args:
            sequence_length (int): length of the sequence to generate

        """
        data = []
        self._sequence_length = sequence_length

        for i in range(self._sequence_length):
            solvent_exposure = self._rng.uniform()
            """The hydrophilicity of that position cannot be inferior than the solvent
            exposure. We want to avoid the not-so-realistic scenario where an hydrophobic
            residue is totally solvent exposed.
            Example:
                solvent_exposure = 0 --> hydrophilicity can be between 0 (hydrophobe) and 1 (polar)
                solvent_exposure = 1 --> Can only be 1 (polar)
            """
            hydrophilicity = self._rng.uniform(low=solvent_exposure)
            # In case the position is totally solvent exposed, the volume won't
            # have much effect in the scoring of that position
            volume = self._rng.uniform()
            data.append((solvent_exposure, hydrophilicity, volume))

        dtype = [('solvent_exposure', 'f4'), ('hydrophilicity', 'f4'), ('volume', 'f4')]
        self._pharmacophore = np.array(data, dtype=dtype)

        # Get the optimal sequence (sequence with the lowest score)
        self._optimal_sequence = _find_optimal_sequence(self._forcefield, self._pharmacophore)

    def generate_pharmacophore_from_sequence(self, peptide_sequence, solvent_exposures=None):
        """Generate a pharmacophore from a peptide sequence

        Args:
            peptide_sequence (str): peptide sequence
            solvent_exposures (array-like): exposure to the solvent for each residue (default: None)

        """
        data = []
        parameters = self._forcefield.parameters()

        if solvent_exposures is not None:
            error_str = "The peptide sequence and solvent exposure have different size."
            assert len(peptide_sequence) == len(solvent_exposures), error_str

        for i, residue_name in enumerate(peptide_sequence):
            param_residue = parameters[parameters['AA1'] == residue_name]

            if solvent_exposures is not None:
                solvent_exposure = solvent_exposures[i]
            else:
                # To avoid hydrophobic residues to be fully solvent exposed
                solvent_exposure = self._rng.uniform(high=param_residue['hydrophilicity'])

            data.append((solvent_exposure, param_residue['hydrophilicity'], param_residue['volume']))

        dtype = [('solvent_exposure', 'f4'), ('hydrophilicity', 'f4'), ('volume', 'f4')]
        self._pharmacophore = np.array(data, dtype=dtype)
        self._optimal_sequence = peptide_sequence

    def generate_random_peptides_from_pharmacophore(self, n=2, sigmas=None):
        """Generate peptide sequences that would not optimally fit the pharmacophore

        Basically we add noise to the current pharmacophore (except solvent exposure)
        and we select the peptide sequence that gives the best score. Maybe not the optimal
        way to generate parents... the other option would be to use a substitution matrix to
        generate parents from optimal_sequence.

        Args:
            n (int): number of peptides to generate
            sigmas (array-like or float): Standard-deviation of the Gaussian noise 
                (default: [0, 0.1, 0.1]; no noise is added to the solvent exposure, only 
                hydrophilicity and volume)

        """
        assert self._pharmacophore is not None, 'Pharmacophore was not generated.'

        parents = []

        if sigmas is None:
            sigmas = [0, 0.1, 0.1]
        else:
            if not isinstance(sigmas, (list, tuple, np.ndarray)):
                sigmas = [sigmas] * len(self._pharmacophore.dtype.names)

            error_str = 'Number of sigma values must be equal to the number of pharmacophore features'
            assert len(sigmas) == len(self._pharmacophore.dtype.names), error_str

        for i in range(n):
            pharmacophore = self._pharmacophore.copy()

            for feature_name, sigma in zip(self._pharmacophore.dtype.names, sigmas):
                if sigma > 0:
                    pharmacophore[feature_name] += self._rng.normal(scale=sigma, size=self._pharmacophore.shape)
                    # Avoid negative numbers
                    pharmacophore[feature_name][pharmacophore[feature_name] < 0.] = 1E-10

            parent = _find_optimal_sequence(self._forcefield, pharmacophore)
            parents.append(parent)

        return parents

    def score_peptides(self, peptides):
        """Score interaction between peptides and the virtual target
        using the provided forcefield

        Args:
            peptides (list): list of peptide strings

        Returns:
            np.ndarray: array of score for each peptide

        """
        score = []

        for peptide in peptides:
            score.append(self._forcefield.score(self._pharmacophore, peptide))

        return np.array(score)

    def export_pharmacophore(self, output_filename):
        """Export virtual pharmacophore as csv file

        Args:
            output_filename (str): output csv filename

        """
        header = ','.join(self._pharmacophore.dtype.names)
        np.savetxt(output_filename, self._pharmacophore, delimiter=',', fmt='%f', header=header)
