#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - virtual target
#

import numpy as np
import pandas as pd


def _find_target_sequence(forcefield, pharmacophore):
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
    target_sequence = ''.join(residue_names[residue_indices])

    return target_sequence


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
        self._dtype = [('solvent_exposure', 'f4'), ('hydrophilicity', 'f4'), 
                       ('volume', 'f4'), ('net_charge', 'i4')]

    def __repr__(self):
        repr_str = 'Pharmacophore:\n'
        repr_str += '%s\n\n' % (pd.DataFrame(self._pharmacophore))
        repr_str += 'Target sequence: %s\n\n' % (self._target_sequence)
        repr_str += 'Parameters for the target sequence:\n'

        parameters = self._forcefield.parameters()
        data = []
        for s in self._target_sequence:
            data.append(parameters[parameters['AA1'] == s])
        repr_str += '%s' % (pd.DataFrame(np.hstack(data)))

        return repr_str

    def load_pharmacophore(self, input_filename):
        """Load the virtual phamacophore

        Args:
            input_filename (str): input csv filename containing the virtual
                pharmacophore

        """
        self._pharmacophore = np.genfromtxt(input_filename, dtype=self._dtype, delimiter=',', skip_header=1)
        # Get the target sequence (sequence with the lowest score)
        self._target_sequence = _find_target_sequence(self._forcefield, self._pharmacophore)

    def target_sequence(self):
        """Return the target sequence for the current pharmacophore

        Returns:
            str: target sequence

        """
        return self._target_sequence

    def generate_random_pharmacophore(self, sequence_length):
        """Generate a random pharmacophore

        Args:
            sequence_length (int): length of the sequence to generate

        """
        parameters = self._forcefield.parameters()

        # Create random sequence
        self._target_sequence = ''.join(self._rng.choice(parameters['AA1'], size=sequence_length))

        # Get the index of each amino acid in the parameters
        indices = [np.argwhere(np.in1d(parameters['AA1'], n)).ravel()[0] for n in self._target_sequence]

        # Retrieve all their parameters
        """The solvent exposure at that position cannot be inferior than the hydrophilicity.
        We want to avoid the not-so-realistic scenario where an hydrophobic residue is totally solvent exposed.
        Example:
            hydrophilicity = 1 --> solvent_exposure can be between 0 (buried) and 1 (in water)
            hydrophilicity = 0 --> Can only be 0 (buried)
        """
        hydrophilicity = parameters[indices]['hydrophilicity']
        volume = parameters[indices]['volume']
        # The pharmacophore needs to be the opposite charge of the target sequence
        net_charge = -1. * parameters[indices]['net_charge']
        solvent_exposure = self._rng.uniform(low=0, high=hydrophilicity, size=sequence_length)

        data = np.stack((solvent_exposure, hydrophilicity, volume, net_charge), axis=-1)
        self._pharmacophore = np.core.records.fromarrays(data.transpose(), dtype=self._dtype)

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

            # The pharmacophore needs to be the opposite charge of the target sequence
            net_charge = -1. * param_residue['net_charge']
            data.append((solvent_exposure, param_residue['hydrophilicity'], param_residue['volume'], net_charge))

        self._pharmacophore = np.array(data, dtype=self._dtype)
        self._target_sequence = peptide_sequence

    def generate_random_peptides_from_pharmacophore(self, n=2, maximum_mutations=3):
        """Generate random peptide sequences from target sequence.

        Args:
            n (int): number of peptides to generate
            maximum_mutations (int): maximum number of mutations (default: 3)

        Returns:
            list: list of mutated peptides

        """
        assert self._pharmacophore is not None, 'Pharmacophore was not generated.'
        assert maximum_mutations <= len(self._target_sequence), 'Max number of mutations greater than peptide length.'

        mutants = []
        parameters = self._forcefield.parameters()

        for i in range(n):
            number_mutations = self._rng.integers(low=1, high=maximum_mutations)
            mutation_positions = self._rng.integers(low=0, high=len(self._target_sequence), size=number_mutations)

            mutant = self._target_sequence.split()

            for mutation_position in mutation_positions:
                mutant[mutation_position] = self._rng.choice(parameters['AA1'])

            mutants.append(''.join(mutants))

        return mutants

    def score_peptides(self, peptides, noise=0):
        """Score interaction between peptides and the virtual target
        using the provided forcefield

        Args:
            peptides (list): list of peptide strings
            noise (float): standard deviation of the Gaussian noise to add (default: 0)

        Returns:
            np.ndarray: array of score for each peptide

        """
        scores = []

        for peptide in peptides:
            score = self._forcefield.score(self._pharmacophore, peptide)

            if noise > 0:
                score += self._rng.normal(scale=noise)
                # Score cannot be negative
                score = 0. if score < 0 else score

            scores.append(score)

        return np.array(scores)

    def export_pharmacophore(self, output_filename):
        """Export virtual pharmacophore as csv file

        Args:
            output_filename (str): output csv filename

        """
        header = ','.join(self._pharmacophore.dtype.names)
        np.savetxt(output_filename, self._pharmacophore, delimiter=',', fmt='%f', header=header)
