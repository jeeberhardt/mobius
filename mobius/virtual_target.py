#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - virtual target
#

import pickle

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
        self._target_sequence = None
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

    @classmethod
    def load_virtual_target(cls, input_filename):
        """Load the virtual target

        Args:
            input_filename (str): serialized (pickle) filename containing the virtual target

        """
        with open(input_filename, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def target_sequence(self):
        """Return the target sequence for the current pharmacophore

        Returns:
            str: target sequence

        """
        return self._target_sequence

    def generate_random_target_sequence(self, sequence_length):
        """Generate a random target sequence and the pharmacophore associated to
        that sequence

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

    def generate_pharmacophore_from_target_sequence(self, target_sequence, solvent_exposures=None):
        """Generate a pharmacophore from a given target sequence

        Args:
            target_sequence (str): peptide sequence
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

    def generate_random_peptides_from_target_sequence(self, n=2, maximum_mutations=3):
        """Generate random peptide sequences from target sequence.

        Args:
            n (int): number of peptides to generate
            maximum_mutations (int): maximum number of mutations (default: 3)

        Returns:
            list: list of mutated peptides

        """
        assert self._target_sequence is not None, 'Target sequence was not generated.'
        assert maximum_mutations <= len(self._target_sequence), 'Max number of mutations greater than the target peptide length.'

        mutants = []
        parameters = self._forcefield.parameters()
        possible_positions = list(range(len(self._target_sequence)))

        for i in range(n):
            number_mutations = self._rng.integers(low=1, high=maximum_mutations)
            mutation_positions = self._rng.choice(possible_positions, size=number_mutations, replace=False)

            mutant = list(self._target_sequence)

            for mutation_position in mutation_positions:
                # This should be replaced by a probability matrix in order to avoid
                # the generation of very far distant sequence...
                mutant[mutation_position] = self._rng.choice(parameters['AA1'])

            mutants.append(''.join(mutant))

        return mutants

    def score_peptides(self, peptides, noise=0):
        """Score interaction between peptides and the pharmacophore
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

    def export_virtual_target(self, output_filename):
        """Export virtual target as serialized (pickle) file

        Args:
            output_filename (str): output pickle filename

        """
        with open(output_filename, 'wb') as w:
            pickle.dump(self, w)
